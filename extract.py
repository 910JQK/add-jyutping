#!/usr/bin/env python3


# ===============================================================
#
# This file is modified from the project
# https://github.com/kerrickstaley/extracting-chinese-subs
#
# which made by Kerrick Staley, and licenced under the MIT License:
# https://github.com/kerrickstaley/extracting-chinese-subs/blob/master/LICENCE
# If you want to do something with this script,
# please read the content of the license carefully.
#
# The original script is also an experimental project,
# so do not use any of these codes in production.
#
# I am not an expert in image processing and do not have
# spare time to learn about knowleges in this field,
# so I directly borrow the code of image processing and ocr
# to my program.
#
# ===============================================================


import inspect
import itertools
import os
import sys
import unicodedata

import cv2
import numpy as np
import pyocr
from PIL import Image

LANG='chi_tra'

class TextExtractor:
  def __init__(self, debug=False):
    self.debug = debug

  def extract(self, img):
    """
    :param numpy.array img: frame of video
    :return str: extracted subtitle text ('' if there is no subtitle)
    """
    self.cleaned = self.clean_image(img)
    self.raw_text = self.run_ocr(self.cleaned)
    return self.post_process_text(self.raw_text)

  def clean_image(self, img):
    """
    :param numpy.array img: frame of video
    :return numpy.array cleaned: cleaned image, ready to run through OCR
    """
    raise NotImplementedError

  def post_process_text(self, text):
    """
    :param str text: text returned by OCR step
    :return str: cleaned text
    """
    if not text:
      return ''

    # hack: tesseract interprets 一 as _
    new_text = [text[0]]
    for before, mid, after in ngroupwise(3, text):
      if mid == '_' and unicodedata.category(before) == unicodedata.category(after) == 'Lo':
        new_text.append('一')
      else:
        new_text.append(mid)
    new_text.append(text[-1])
    txt = ''.join(new_text)

    # strip out non-Chinese characters
    rv = []
    for c in txt:
      if unicodedata.category(c) != 'Lo':
        continue
      rv.append(c)

    return ''.join(rv)

  def run_ocr(self, img):
    """
    :param numpy.array img: cleaned image
    :return str: extracted subtitle text ('' if there is no subtitle)
    """
    # average character is 581 pixels
    if np.count_nonzero(img) < 1000:
      return ''

    tool = pyocr.get_available_tools()[0]
    pil_img = Image.fromarray(img)
    return tool.image_to_string(
        pil_img,
        lang=LANG,
      )


class E0(TextExtractor):
  top = 590
  bottom = 650
  left = 250  # min observed was 300 pixels in, each char is 50 pixels wide
  right = 1030  # max observed was 300 pixels in from the right

  def clean_image(self, img):
    cropped = img[
        self.top: self.bottom,
        self.left: self.right]
    return self.clean_after_crop(cropped)

  def clean_after_crop(self, cropped):
    img = threshold(cropped)
    img = dilate_erode3(img)
    img = dilate3(img)
    img = img & dilate_erode5(cv2.Canny(cropped, 400, 600))
    return img


class E1(E0):
  def get_canny_mask(self, cropped):
    mask = cv2.Canny(cropped, 400, 600)
    mask = dilate(mask, 5)
    mask = erode(mask, 5)
    return mask

  def sharpen(self, img):
    blurred = cv2.GaussianBlur(img, (3, 3), 0)
    return cv2.addWeighted(img, 2, blurred, -1, 0)

  def clean_after_crop(self, cropped):
    self.sharpened = img = self.sharpen(cropped)
    if self.debug:
      show_image(self.sharpened)
    self.thresholded = img = threshold(img, min_value=191)
    if self.debug:
      show_image(self.thresholded)
    self.canny_mask = self.get_canny_mask(cropped)
    img = img & self.canny_mask
    if self.debug:
      show_image(self.canny_mask)
      show_image(img)
    img = remove_small_islands(img)
    img = dilate3(img)
    return img


class E2(E1):
  def get_border_floodfill_mask(self):
    mask = np.zeros(self.thresholded.shape)
    _, contours, hierarchy = cv2.findContours(self.thresholded, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    for root_idx, contour in enumerate(contours):
      left, top, width, height = cv2.boundingRect(contour)
      right = left + width
      bottom = top + height
      if not (top <= 4 or left <= 4
              or bottom >= self.thresholded.shape[0] - 5 or right >= self.thresholded.shape[1] - 5):
        continue

      cv2.fillPoly(mask, pts=[contour], color=(255, 255, 255))
      for child_contour, (_, _, _, parent_idx) in zip(contours, hierarchy[0]):  # TODO no idea why we have to do [0]
        if parent_idx != root_idx:
          continue
        cv2.fillPoly(mask, pts=[child_contour], color=(0, 0, 0))

    # because we do a dilate3 in super().clean_after_crop, we also need to do that here so the mask matches when we
    # subtract
    mask = dilate(mask, 3)

    return mask

  def clean_after_crop(self, cropped):
    img = super().clean_after_crop(cropped)
    self.border_floodfill_mask = self.get_border_floodfill_mask()
    if self.debug:
      show_image(self.border_floodfill_mask)
    return img - self.get_border_floodfill_mask()


class E3(E2):
  def get_border_floodfill_mask(self):
    h, w = self.thresholded.shape[:2]
    mask = np.zeros((h + 2, w + 2), np.uint8)

    border_points = []
    for r in range(5):
      for c in range(w):
        # top border
        border_points.append((r, c))
        # bottom border
        border_points.append((h - 1 - r, c))
    for c in range(5):
      for r in range(h):
        # left border
        border_points.append((r, c))
        # right border
        border_points.append((r, w - 1 - c))

    for r, c in border_points:
        if not self.thresholded[r][c]:
          continue
        # The (255 << 8) incantation means set mask value to 255 when filling. The | 8 means do an 8-neighbor fill.
        cv2.floodFill(self.thresholded, mask, (c, r), 255, flags=(255 << 8) | cv2.FLOODFILL_MASK_ONLY | 8)

    # because we do a dilate3 in super().clean_after_crop, we also need to do that here so the mask matches when we
    # subtract
    mask = dilate(mask, 3)

    return mask[1:-1, 1:-1]


class E4(E3):
  def get_canny_mask(self, cropped):
    mask = cv2.Canny(cropped, 400, 600)
    mask = dilate(mask, 5)
    mask = erode(mask, 3)
    return mask


class E5(E3):
  def sharpen(self, img):
    blurred = cv2.GaussianBlur(img, (3, 3), 0)
    return cv2.addWeighted(img, 2.7, blurred, -1.7, 0)


class B0(E0):
  """
  The first model I blogged about (in the Part 1 article).

  Pass rate: 18%.
  """

  def clean_after_crop(self, cropped):
    img = cv2.inRange(cropped, (200, 200, 200), (255, 255, 255))
    return img


class B1(B0):
  """
  Thresholding using HSV.

  Pass rate: 26%.
  """

  def clean_after_crop(self, cropped):
    return threshold(cropped, min_value=180, max_saturation=30)


class B2(B1):
  """
  Dilating the output of B1.

  Pass rate: 52%.
  """

  def clean_after_crop(self, cropped):
    return dilate(super().clean_after_crop(cropped), 3)


def ngroupwise(n, iterable):
  # generalization of the "pairwise" recipe
  iterators = list(itertools.tee(iterable, n))
  for i in range(n):
    for j in range(i):
      next(iterators[i], None)

  return zip(*iterators)


def threshold(img, min_value=170, max_saturation=25):
  hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
  return cv2.inRange(hsv, (0, 0, min_value), (179, max_saturation, 255))


def dilate_erode5(img):
  "Closes the img"
  kernel = np.ones((5, 5), np.uint8)
  img = cv2.dilate(img, kernel)
  img = cv2.erode(img, kernel)
  return img


def dilate_erode3(img):
  "Closes the img"
  kernel = np.ones((3, 3), np.uint8)
  img = cv2.dilate(img, kernel)
  img = cv2.erode(img, kernel)
  return img


def dilate3(img):
  kernel = np.ones((3, 3), np.uint8)
  return cv2.dilate(img, kernel)


def dilate(img, n=3):
  kernel = np.ones((n, n), np.uint8)
  return cv2.dilate(img, kernel)


def erode(img, n=3):
  kernel = np.ones((n, n), np.uint8)
  return cv2.erode(img, kernel)


def remove_small_islands(img, min_pixels=2):
  mask = np.zeros(img.shape)
  im2, contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
  for contour in contours:
    if cv2.contourArea(contour) < min_pixels:
      cv2.fillPoly(mask, pts=contour, color=(255, 255, 255))
  return img - mask


def show_image(img):
  # compute the name of the object we're displaying
  var_name = None
  lcls = inspect.stack()[1][0].f_locals
  if 'self' in lcls:
    for k, v in lcls['self'].__dict__.items():
      if id(img) == id(v):
        var_name = 'self.' + k
        break

  if var_name is None:
    for name in lcls:
      if name == '_':
        continue
      if id(img) == id(lcls[name]):
        var_name = name
        break

  if var_name is None:
    var_name = '(unknown image)'

  # resize image
  scale_factor = 4
  img = cv2.resize(img, (0, 0), None, scale_factor, scale_factor, cv2.INTER_NEAREST)

  cv2.imshow(var_name, img)
  while True:
    key = cv2.waitKey(0)
    if key == ord('q'):
      raise Exception('quitting')
    if ord(' ') <= key <= ord('~'):
      break
  cv2.destroyAllWindows()


def pad_string(s, l):
  chars_taken = len(s)
  for c in s:
    if unicodedata.east_asian_width(c) == 'W':
      chars_taken += 1

  return s + ' ' * (l - chars_taken)




