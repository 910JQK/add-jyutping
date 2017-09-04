#!/usr/bin/env python3


# ===============================================================
#
# Script to add Jyutping to TVB Programs
#
# This is an experimental script. Do not use it in production.
#
# usage: jyutping.py [-h] [--top TOP] [--bottom BOTTOM] [--left LEFT]
#                    [--right RIGHT]
#                    video_file
# The four parameters top, bottom, left and right is the edge
# of the rectangle that contains the subtitle.
# The default values are set for The News at Six Thrity.
#
# Output is the .srt format file content and some progress info.
# If you want to generate a .srt subtitle file, please
# redirect the standard output to a .srt file or
# redirect the standard error ouput to the null device.
#
# Example:
#
#     $ ./jyutping.py xxx.mp4 > xxx-jyutping.srt
#
# The content of this script (only this file)
# is licenced under the Unlicense (http://unlicense.org/).
# In other words, this file is released into the public domain.
#
# ===============================================================


import os
import cv2
import sys
import json
from math import floor
from extract import E3 as Extractor
from argparse import ArgumentParser


INTERVAL = 20
TIME_PER_CHAR = 0.2
SCRIPT_PATH = os.path.dirname(os.path.realpath(__file__))
DICT = json.loads(open(SCRIPT_PATH + '/data.json', 'r').read())
count = 0

parser = ArgumentParser(description='Add Jyutping for TVB Programs (Output .srt file content to standard output)')
parser.add_argument('--top', type=int, default=590,
                    help='top edge of subtitle rect')
parser.add_argument('--bottom', type=int, default=650,
                    help='bottom edge of subtitle rect')
parser.add_argument('--left', type=int, default=250,
                    help='left edge of subtitle rect')
parser.add_argument('--right', type=int, default=1030,
                    help='right edge of subtitle rect')
parser.add_argument('video_file')

def msg(*args, **kwargs):
    print(*args, **kwargs, file=sys.stderr)


def sec_to_str(total_sec):
    def add_zero(n):
        if n < 10:
            return '0' + str(n)
        else:
            return str(n)
    hours = floor(total_sec / 3600)    
    minutes = floor(total_sec % 3600 / 60)
    seconds = floor(total_sec % 60)
    milliseconds = ((total_sec % 60) - seconds)*1000
    h_str = add_zero(hours)
    m_str = add_zero(minutes)
    s_str = add_zero(seconds)
    milli_str = str(round(milliseconds))
    milli_str += (3-len(milli_str))*'0'
    return '%s:%s:%s,%s' % (h_str, m_str, s_str, milli_str)


def add_jyutping(string):
    result = ''
    for char in string:
        if DICT.get(char):
            result += (char + DICT[char])
        else:
            result += char
        result += ' '
    return result


def output_subtitle(total_sec, text, dur):
    global count
    print(
'''%d
%s --> %s
%s
        
''' % (
            count,
            sec_to_str(total_sec),
            sec_to_str(total_sec + dur),
            add_jyutping(text)
        )
    )
    sys.stdout.flush()
    count += 1


def similarity(text1, text2):
    l1 = len(text1)
    l2 = len(text2)
    if abs(l1-l2) > 3:
        return 0
    overlap = 0
    chars = {}
    for char in text1:
        chars[char] = True
    for char in text2:
        if chars.get(char):
            overlap += 1
    return overlap / ((l1+l2)/2)


def main():
    args = parser.parse_args()
    extractor = Extractor()
    extractor.top = args.top
    extractor.bottom = args.bottom
    extractor.left = args.left
    extractor.right = args.right
    file_name = args.video_file
    video = cv2.VideoCapture(file_name)
    fps = video.get(cv2.CAP_PROP_FPS)
    length = video.get(cv2.CAP_PROP_FRAME_COUNT)
    interval_sec = INTERVAL / fps
    success = True
    index = 0
    characters = 0
    last_text = ''
    last_sec = 0
    while success:
        success, frame = video.read()
        total_sec = index / fps
        if index % INTERVAL == 0 and total_sec >= last_sec:
            text = extractor.extract(frame)
            characters += len(text)
            if len(text) == 0:
                dur = total_sec - last_sec
                output_subtitle(last_sec, last_text, dur)
            else:
                if similarity(text, last_text) > 0.6:
                    dur = interval_sec / 2
                else: 
                    dur = len(text)*TIME_PER_CHAR                   
                output_subtitle(last_sec, text, dur)
            msg(
                "%d%% Processed, %d Characters Scanned"
                % (floor((index / length)*100), characters)
            )
            last_sec = last_sec + dur
            last_text = text
        index += 1


if __name__ == '__main__':
    main()
