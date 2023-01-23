import numpy as np
from PIL import Image
import time
import numba as nb
from numba import jit
from numba import njit
from numba import prange
import matplotlib.pyplot as plt
import cv2


def numb_create_mask(mask, image, lower_color, upper_color):

 for i in range(0,image.shape[0]):
    for j in range(0,image.shape[1]):
        if np.all(image[i][j] <= upper_color) & np.all(image[i][j] >= lower_color):
            mask[i][j] = 255
        else:
            mask[i][j] = 0

 return mask
numb_create_mask = jit(numb_create_mask)

def create_mask(mask, image, lower_color, upper_color):

  mask[np.all(image <= upper_color,axis=2) & np.all(image >= lower_color,axis=2)] = 255
  mask[np.all(image > upper_color,axis=2) & np.all(image < lower_color,axis=2)] = 0

  return mask

def add_background(bg, im):
    image = np.array(im)
    background = np.array(bg)

    lower_color = np.array([0, 100, 0])
    upper_color = np.array([130, 255, 90])

    mask = np.random.randint(0, 1, size=(image.shape[0], image.shape[1]))
    mask = create_mask(mask, image, lower_color, upper_color)

    image[mask != 0] = [0, 0, 0]

    background = background[0:720, 0:1280]
    background[mask == 0] = [0, 0, 0]

    final_image = background + image

    return final_image

def numba_add_background(bg, im):
    image = np.array(im)
    background = np.array(bg)

    lower_color = np.array([0, 100, 0])
    upper_color = np.array([130, 255, 90])

    mask = np.random.randint(0, 1, size=(image.shape[0], image.shape[1]))
    numb_mask = numb_create_mask(mask, image, lower_color, upper_color)

    image[mask != 0] = [0, 0, 0]

    background = background[0:720, 0:1280]
    background[mask == 0] = [0, 0, 0]

    final_image = background + image

    return final_image

def add_background_fast(background, image):
    image = np.array(image)
    l_green = np.array([0, 100, 0])
    u_green = np.array([120, 255, 100])

    mask = cv2.inRange(image, l_green, u_green)
    res = cv2.bitwise_and(image, image, mask=mask)

    final_image = image - res
    final_image = np.where(final_image == 0, background, final_image)

    return final_image
