
import os
import os.path as osp
import argparse
import numpy as np
import soundfile as sf
from moviepy.editor import AudioFileClip, VideoFileClip
from moviepy.audio.AudioClip import AudioArrayClip

import logging
logging.getLogger('xformers').setLevel(logging.ERROR) # shutup triton, before torch
logging.getLogger('diffusers.models.modeling_utils').setLevel(logging.CRITICAL)

import torch
import torchvision

from huggingface_hub import snapshot_download
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

from foleycrafter.models.onset import torch_utils
from foleycrafter.models.time_detector.model import VideoOnsetNet
from foleycrafter.pipelines.auffusion_pipeline import Generator, denormalize_spectrogram
from foleycrafter.utils.util import build_foleycrafter, read_frames_with_moviepy
from audio_utils import normalize_audio
logging.getLogger('diffusers').setLevel(logging.ERROR)

from utils import vid_list, progbar, basename, split

parser = argparse.ArgumentParser()
parser.add_argument('-i',  '--input',   default="_in", help="input video folder path")
parser.add_argument('-o',  '--save_dir', default='_out')
parser.add_argument('-t',  '--prompt',  default="", help="text prompt for audio generation")
parser.add_argument('-un', '--nprompt',default="", help="negative prompt for audio generation")
parser.add_argument('-md', '--ckpt',    default='models', help='Main models directory')
parser.add_argument('-s',  '--steps',   default=23, type=int, help="number of diffusion steps")
parser.add_argument('-cut','--maxcut',  default=99999, type=float, help="Maximum continuous length in seconds; split to pieces if longer")
parser.add_argument("--semantic_scale", default=0.8, type=float, help="visual content scale") # 1.
parser.add_argument("--temporal_scale", default=0.3, type=float, help="temporal align scale") # .2
parser.add_argument('-S',  "--seed",    default=None, type=int, help="ramdom seed")
parser.add_argument('-v',  '--verbose', action='store_true')
a = parser.parse_args()

device = torch.device('cuda')

def build_models():
    if not osp.isfile(osp.join(a.ckpt, "temporal_adapter.ckpt")):
        snapshot_download("ymzhang319/FoleyCrafter", local_dir=a.ckpt)
    if not osp.isdir(osp.join(a.ckpt, "auffusion")):
        snapshot_download("auffusion/auffusion-full-no-adapter", local_dir=osp.join(a.ckpt, "auffusion"))
        
    vocoder = Generator.from_pretrained(a.ckpt, subfolder="vocoder").to(device)
    # load time_detector
    time_detector_ckpt = osp.join(osp.join(a.ckpt, "timestamp_detector.pth.tar"))
    time_detector = VideoOnsetNet(False)
    time_detector, _ = torch_utils.load_model(time_detector_ckpt, time_detector, device=device, strict=True)
    # load adapters
    pipe = build_foleycrafter(osp.join(a.ckpt, 'auffusion')).to(device)
    ckpt = torch.load(osp.join(a.ckpt, "temporal_adapter.ckpt"))

    # load temporal adapter
    if "state_dict" in ckpt.keys():
        ckpt = ckpt["state_dict"]
    gligen_ckpt = {}
    for key, value in ckpt.items():
        if key.startswith("module."):
            gligen_ckpt[key[len("module.") :]] = value
        else:
            gligen_ckpt[key] = value
    pipe.controlnet.load_state_dict(gligen_ckpt, strict=False)

    # load semantic adapter
    pipe.load_ip_adapter(osp.join(a.ckpt, "semantic"), subfolder="", weight_name="semantic_adapter.bin", image_encoder_folder=None)
    ip_adapter_weight = a.semantic_scale
    pipe.set_ip_adapter_scale(ip_adapter_weight)

    img_procsr = CLIPImageProcessor()
    # img_encoder = CLIPVisionModelWithProjection.from_pretrained(a.ckpt, subfolder="image").to(device)
    img_encoder = CLIPVisionModelWithProjection.from_pretrained("h94/IP-Adapter", subfolder="models/image_encoder").to(device)

    return pipe, vocoder, time_detector.to(device), img_encoder, img_procsr


@torch.no_grad()
def main():
    os.makedirs(a.save_dir, exist_ok=True)

    vision_transform_list = [
        torchvision.transforms.Resize((128, 128)),
        torchvision.transforms.CenterCrop((112, 112)),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
    video_transform = torchvision.transforms.Compose(vision_transform_list)

    pipe, vocoder, time_detector, img_encoder, img_procsr = build_models()
    sr = 16000

    input_list = vid_list(a.input) if osp.isdir(a.input) else [a.input]
    assert len(input_list) != 0, "input directory is empty!"
    g_ = torch.Generator(device=device)
    if a.seed is not None: g_ = g_.manual_seed(a.seed)
    pbar = progbar(len(input_list))
    for input_video in input_list:
        frames, duration, fps = read_frames_with_moviepy(input_video) # [n,h,w,c]
        audioslices = []
        slices = split(frames, a.maxcut * fps)
        pbar2 = progbar(len(slices))
        for slice in slices:
            frameslice = frames[slice]
            curdur = len(frameslice) / fps
            W = int(curdur * 1024 / 10) # length of sound in centi-seconds

            time_frames = video_transform(torch.FloatTensor(frameslice).to(device).permute(0,3,1,2)).unsqueeze(0).permute(0,2,1,3,4)
            preds = time_detector({"frames": time_frames})
            preds = torch.sigmoid(preds) # [1,x]
            time_condition = [-1 if preds[0][int(fps * i * 10 / 1024)] < 0.5 else 1 for i in range(W)]
            if W < 1024:
                time_condition = time_condition + [-1] * (1024 - W)
                W = 1024
            time_condition = torch.FloatTensor(time_condition)[None, None, None].repeat(1, 1, 256, 1) # w -> b c h w

            images  = img_procsr(images = frameslice[::len(frameslice)//10], return_tensors="pt").to("cuda") # take 10 frames
            del frameslice, time_frames, preds; torch.cuda.empty_cache()
            img_emb = img_encoder(**images).image_embeds.mean(dim=0, keepdim=True)[None, None]
            img_emb = torch.cat([torch.zeros_like(img_emb), img_emb], dim=1) # [1,2,1,1024] average emb for all frames
            del images; torch.cuda.empty_cache()

            output = pipe(
                prompt = a.prompt,
                negative_prompt = a.nprompt,
                ip_adapter_image_embeds = img_emb,
                image = time_condition,
                # audio_length_in_s=10,
                controlnet_conditioning_scale = a.temporal_scale,
                num_inference_steps = a.steps,
                height = 256,
                width = W,
                output_type = "pt",
                generator = g_,
                # guidance_scale=0,
            )

            audio = denormalize_spectrogram(output.images[0]) # [1,3,256,W] -> [256,W]
            audio = vocoder.inference(audio)[:, :int(curdur*sr)] # [1,x]
            audioslices += [audio]
            pbar2.upd(uprows=1)

        audio = torch.cat(audioslices, dim=1)
        audio = normalize_audio(audio, sr).cpu().numpy() # [-1..1]
        outpath = osp.join(a.save_dir, basename(input_video))
        video = VideoFileClip(input_video)
        # sf.write(outpath + '.wav', (audio * 32768).astype("int16")[0], sr)
        # video.audio = AudioFileClip(outpath + '.wav') # .subclip(0, duration)
        video.audio = AudioArrayClip(audio.transpose().repeat(2, 1).astype("float32"), fps=sr)
        video.write_videofile(outpath + '.mp4', verbose=False, logger=None)
        pbar.upd(uprows=2)


if __name__ == "__main__":
    main()
