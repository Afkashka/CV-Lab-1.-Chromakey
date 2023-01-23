import cv2
import chrom_script as cs
from PIL import Image
import numpy as np
import time
from matplotlib import pyplot as plt

cap = cv2.VideoCapture('cat green screen.mp4')
if cap.isOpened() == False:
    print('Не возможно открыть файл')

back = Image.open("background.jpg")
flag = False

while cap.isOpened():
    fl, img = cap.read()
    if img is None:
        break
    if flag:
        #new_im = cs.numba_add_background(back,img)
        #new_im = cs.add_background_fast(back,img)
        new_im = cs.add_background(back,img)
        cv2.imshow("Cat", new_im)
    else:
        cv2.imshow("Cat", img)
    if cv2.waitKey(25) == ord('q'):
        break
    if cv2.waitKey(25) == ord('e'):
        flag = not flag
cap.release()
cv2.destroyAllWindows()


img = Image.open("cat green.jpg")
start_time = time.time()
i1 = cs.numba_add_background(back,img)
print("Время работы функции с numba: %s секунд" % (time.time() - start_time))


start_time = time.time()
i2 = cs.add_background(back,img)
print("Время работы функции без numba: %s секунд" % (time.time() - start_time))



start_time = time.time()
i3 = cs.add_background_fast(back,img)
print("Время работы библиотечной функции: %s секунд" % (time.time() - start_time))
