
import os
import sys

def basename(file):
    return os.path.splitext(os.path.basename(file))[0]

def vid_list(path, subdir=None):
    if subdir is True:
        files = [os.path.join(dp, f) for dp, dn, fn in os.walk(path) for f in fn]
    else:
        files = [os.path.join(path, f) for f in os.listdir(path)]
    files = [f for f in files if os.path.splitext(f.lower())[1][1:] in ['mov', 'avi', 'mp4']]
    return sorted([f for f in files if os.path.isfile(f)])

def split(x, maxlen=20, span=None, dim=0):
    span = maxlen if span is None else min(span, maxlen)
    length = len(x) if isinstance(x, list) else x.shape[dim]
    if length <= maxlen: return [slice(0, length)]
    num = int(max(2, (length+span-1 if span==maxlen else length) // span)) # fit span size or go above
    lens = [length // num + (1 if i < length % num else 0) for i in range(num)]
    slices = [slice(sum(lens[:i]), sum(lens[:i]) + size) for i, size in enumerate(lens)]
    # ids = [list(range(s.start, s.stop)) for s in slices]
    # chunks = [x[s] for s in slices] # if isinstance(x, list) else x.narrow(dim, s.start, s.stop - s.start)
    return slices
    
# # # = = = progress bar = = = # # #

import time
from shutil import get_terminal_size
import ipywidgets as ipy
import IPython

try: # notebook/colab
    get_ipython().__class__.__name__
    maybe_colab = True
except: # normal console
    maybe_colab = False

class ProgressIPy(object):
    def __init__(self, task_num=10, start_num=0, start=True):
        self.task_num = task_num - start_num
        self.pbar = ipy.IntProgress(min=0, max=self.task_num, bar_style='') # (value=0, min=0, max=max, step=1, description=description, bar_style='')
        self.labl = ipy.Label()
        IPython.display.display(ipy.HBox([self.pbar, self.labl]))
        self.completed = 0
        self.start_num = start_num
        if start:
            self.start()

    def start(self, task_num=None, start_num=None):
        if task_num is not None:
            self.task_num = task_num
        if start_num is not None:
            self.start_num = start_num
        self.labl.value = '{}/{}'.format(self.start_num, self.task_num + self.start_num)
        self.start_time = time.time()

    def upd(self, *p, **kw):
        self.completed += 1
        elapsed = time.time() - self.start_time + 0.0000000000001
        fps = self.completed / elapsed if elapsed>0 else 0
        if self.task_num > 0:
            finaltime = time.asctime(time.localtime(self.start_time + self.task_num * elapsed / float(self.completed)))
            fin = ' end %s' % finaltime[11:16]
            percentage = self.completed / float(self.task_num)
            eta = int(elapsed * (1 - percentage) / percentage + 0.5)
            self.labl.value = '{}/{}, rate {:.3g}s, time {}s, left {}s, {}'.format(self.completed + self.start_num, self.task_num + self.start_num, 1./fps, shortime(elapsed), shortime(eta), fin)
        else:
            self.labl.value = 'completed {}, time {}s, {:.1f} steps/s'.format(self.completed + self.start_num, int(elapsed + 0.5), fps)
        self.pbar.value += 1
        if self.completed == self.task_num: self.pbar.bar_style = 'success'
        return self.completed

    def reset(self, start_num=0, task_num=None):
        self.start_time = time.time()
        self.start_num = start_num
        if task_num is not None:
            self.task_num = task_num

class ProgressBar(object):
    '''A progress bar which can print the progress
    modified from https://github.com/hellock/cvbase/blob/master/cvbase/progress.py
    '''
    def __init__(self, count=0, start_num=0, bar_width=50, start=True):
        self.task_num = count - start_num
        max_bar_width = self._get_max_bar_width()
        self.bar_width = (bar_width if bar_width <= max_bar_width else max_bar_width)
        self.completed = 0
        self.start_num = start_num
        if start:
            self.start()

    def _get_max_bar_width(self):
        terminal_width, _ = get_terminal_size()
        max_bar_width = min(int(terminal_width * 0.6), terminal_width - 50)
        if max_bar_width < 10:
            print('terminal is small ({}), make it bigger for proper visualization'.format(terminal_width))
            max_bar_width = 10
        return max_bar_width

    def start(self, task_num=None, start_num=None):
        if task_num is not None:
            self.task_num = task_num
        if start_num is not None:
            self.start_num = start_num
        if self.task_num > 0:
            sys.stdout.write('[{}] {}/{} \n{}\n'.format(' ' * self.bar_width, self.start_num, self.task_num + self.start_num, 'Start...'))
        else:
            sys.stdout.write('completed: 0, elapsed: 0s')
        sys.stdout.flush()
        self.start_time = time.time()

    def upd(self, msg=None, uprows=0):
        self.completed += 1
        elapsed = time.time() - self.start_time + 0.0000000000001
        fps = self.completed / elapsed if elapsed>0 else 0
        if self.task_num > 0:
            percentage = self.completed / float(self.task_num)
            eta = int(elapsed * (1 - percentage) / percentage + 0.5)
            finaltime = time.asctime(time.localtime(self.start_time + self.task_num * elapsed / float(self.completed)))
            fin_msg = ' %ss left, end %s' % (shortime(eta), finaltime[11:16])
            if msg is not None: fin_msg += '  ' + str(msg)
            mark_width = int(self.bar_width * percentage)
            bar_chars = 'X' * mark_width + '-' * (self.bar_width - mark_width) # - - -
            sys.stdout.write('\033[%dA' % (uprows+2)) # cursor up 2 lines + extra if needed
            sys.stdout.write('\033[J')  # clean the output (remove extra chars since last display)
            try:
                sys.stdout.write('[{}] {}/{}, rate {:.3g}s, time {}s, left {}s \n{}\n'.format(
                    bar_chars, self.completed + self.start_num, self.task_num + self.start_num, 1./fps, shortime(elapsed), shortime(eta), fin_msg))
            except:
                sys.stdout.write('[{}] {}/{}, rate {:.3g}s, time {}s, left {}s \n{}\n'.format(
                    bar_chars, self.completed + self.start_num, self.task_num + self.start_num, 1./fps, shortime(elapsed), shortime(eta), '<< unprintable >>'))
        else:
            sys.stdout.write('completed {}, time {}s, {:.1f} steps/s'.format(self.completed, int(elapsed + 0.5), fps))
        sys.stdout.flush()

    def reset(self, start_num=0, count=None, newline=False):
        self.start_time = time.time()
        self.start_num = start_num
        if count is not None:
            self.task_num = count - start_num
        if newline is True:
            sys.stdout.write('\n\n')

progbar = ProgressIPy if maybe_colab else ProgressBar

def time_days(sec):
    return '%dd %d:%02d:%02d' % (sec/86400, (sec/3600)%24, (sec/60)%60, sec%60)
def time_hrs(sec):
    return '%d:%02d:%02d' % (sec/3600, (sec/60)%60, sec%60)
def shortime(sec):
    if sec < 60:
        time_short = '%d' % (sec)
    elif sec < 3600:
        time_short  = '%d:%02d' % ((sec/60)%60, sec%60)
    elif sec < 86400:
        time_short  = time_hrs(sec)
    else:
        time_short = time_days(sec)
    return time_short

