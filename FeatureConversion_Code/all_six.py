import numpy as np
#import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image
#import keras
import glob
import pywt
from cv2 import cv2
import one2six

import os

def mean(image):
    img = image
    #img = cv2.resize(img, (500, 400))
    _, threshold = cv2.threshold(img, 155, 255, cv2.THRESH_BINARY)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mean_c = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, 12)
    gaus = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 91, 12)
    # cv2.imshow("Img", img)
    # cv2.imshow("Binary threshold", threshold)
    # cv2.imshow("Mean C", mean_c)
    # cv2.imshow("Gaussian", gaus)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    img2 = np.zeros_like(img)
    img2[:, :, 0] = gaus
    img2[:, :, 1] = gaus
    img2[:, :, 2] = gaus
    #cv2.imwrite("foo/image_1_m.jpg", img2)
    #print(img2.shape)
    return img2

def w2d(img, mode='haar', level=1):
    #imArray = cv2.imread(img)
    #imArray = cv2.resize(img, (500, 400))
    #Datatype conversions
    #gaussian_1 = cv2.GaussianBlur(imArray, (9, 9), 10.0)
    #imArray = cv2.addWeighted(imArray, 2.5, gaussian_1, -0.5, 0, imArray)
    #convert to grayscale
    imArray = cv2.cvtColor( img,cv2.COLOR_RGB2GRAY )
    #convert to float
    imArray =  np.float32(imArray)
    imArray /= 255;
    # compute coefficients
    coeffs=pywt.wavedec2(imArray, mode, level=level)

    #Process Coefficients
    coeffs_H=list(coeffs)
    coeffs_H[0] *= 0;

    # reconstruction
    imArray_H=pywt.waverec2(coeffs_H, mode);
    imArray_H *= 255;
    imArray_H =  np.uint8(imArray_H)
    #Display result
    #cv2.imshow('image',imArray_H)
    img2 = np.zeros_like(img)
    img2[:, :, 0] = imArray_H
    img2[:, :, 1] = imArray_H
    img2[:, :, 2] = imArray_H
    #cv2.imwrite("foo/image_1_w.jpg", img2)
    #print(img2.shape)

    #height, width = imArray_H.shape[:2]
    #print(np.shape(imArray_H))
    #print(width)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    return img2

def corner(image):

    img=image.copy()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    gray = np.float32(gray)
    dst = cv2.cornerHarris(gray, 2, 3, 0.04)

    # result is dilated for marking the corners, not important
    dst = cv2.dilate(dst, None)

    # Threshold for an optimal value, it may vary depending on the image.
    img[dst > 0.01 * dst.max()] = [0, 0, 255]

    #cv2.imshow('dst', image)
    #cv2.imwrite("foo/image_1_c.jpg", img)
    #print(img.shape)
    return img

def build_filters():
    filters = []
    ksize = 31
    for theta in np.arange(0, np.pi, np.pi / 16):
        kern = cv2.getGaborKernel((ksize, ksize), 4.0, theta, 10.0, 0.5, 0, ktype=cv2.CV_32F)
    kern /= 1.5 * kern.sum()
    filters.append(kern)
    return filters


def process(img, filters):
    accum = np.zeros_like(img)
    for kern in filters:
        fimg = cv2.filter2D(img, cv2.CV_8UC3, kern)
    np.maximum(accum, fimg, accum)
    return accum

def gabor(image):
    filters = build_filters()
    res1 = process(image, filters)
    #cv2.imshow('result', res1)
    #cv2.imwrite("foo/image_1_g.jpg", res1)
    #print(res1.shape)
    return res1

def whiteBackground(image):
    im = image
    data = np.array(im)
    r1, g1, b1 = 0, 0, 0  # Original value
    r2, g2, b2 = 255, 255, 255  # Value that we want to replace it with
    red, green, blue = data[:, :, 0], data[:, :, 1], data[:, :, 2]
    mask = (red == r1) & (green == g1) & (blue == b1)
    data[:, :, :3][mask] = [r2, g2, b2]
    im = Image.fromarray(data)
    im.save('fig1_modified.png')
    im = np.array(im)
    return im

def createMask2(image):
    pic = image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edged = cv2.Canny(gray, 50, 100)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)
    (_, cnts, _) = cv2.findContours(closed.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.ones(image.shape[:2], dtype="uint8") * 255
    for c in cnts:
        cv2.drawContours(mask, [c], -1, 0, -1)
    mask_inv = cv2.bitwise_not(mask)
    image = cv2.bitwise_and(image, image, mask=mask)
    image2 = cv2.bitwise_and(pic, pic, mask=mask_inv)
    return whiteBackground(image2)


# Loop for readin images from class label folder one by one and then will create a training/test set 
# with all class label folder and One to SIX featured image for each image 
fruit_images = []
labels = []
#for fruit_dir_path in glob.glob("Test/check/*"):#Select this line for Test image conversion
for fruit_dir_path in glob.glob("Train/check/*"):#Select this line for Train image conversion
    i=0
    fruit_label = fruit_dir_path.split("/")[-1]
    #drn = 'F:/python/Testing/One2Six/six/Test/result/' + fruit_label
    drn = 'result/' + fruit_label  #Create a file name 'check' in result folder (if not already)
    os.mkdir(drn)
    for image_path in glob.glob(os.path.join(fruit_dir_path, "*.jpg")):
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)

        #image = cv2.resize(image, (45, 45))
        #image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        #image2 = createMask2(image)
        image2 = image
        gaussian_1 = cv2.GaussianBlur(image2, (9, 9), 10.0)
        Image3 = cv2.addWeighted(image2, 1.5, gaussian_1, -0.5, 0, image2)
        fruit_images.append(Image3)
        labels.append(fruit_label)
        print(fruit_label)
        # drn='result/'+fruit_label
        # os.mkdir(drn)
        #RGB -> The actual image Featured Image 1
        cv2.imwrite(os.path.join(drn +"/image_" +str(i)+ '.jpg'), Image3)
        # Gabor -> The Gabor feature image Featured Image 2
        gab = gabor(Image3)
        #cv2.imshow("f",gab)
        #cv2.waitKey(0)
        cv2.imwrite(os.path.join(drn + "/image_" + str(i) + '_g.jpg'), gab)
        #Corner -> The image after corner detection Featured Image 3
        cor = corner(Image3)
        cv2.imwrite(os.path.join(drn + "/image_" + str(i) + '_c.jpg'), cor)
        #Wevelet -> The image after weblet conversion  Featured Image 4
        wev = w2d(image2, 'db1', 10)
        cv2.imwrite(os.path.join(drn + "/image_" + str(i) + '_w.jpg'), wev)
        #Gaussian_Mean -> The image after Gaussian_Mean Featured Image 5
        men = mean(Image3)
        cv2.imwrite(os.path.join(drn + "/image_" + str(i) + '_m.jpg'), men)
        #LAB -> The image after LAB color conversion Featured Image 6
        brightLAB = cv2.cvtColor(Image3, cv2.COLOR_BGR2LAB)
        cv2.imwrite(os.path.join(drn + "/image_" + str(i) + '_l.jpg'), brightLAB)
        i=i+1
fruit_images = np.array(fruit_images)
labels = np.array(labels)