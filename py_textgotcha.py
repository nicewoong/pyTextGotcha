#!/usr/bin/python
# -*- coding: utf-8 -*-
"""  Detect text in the image.
And processes the image to extract the text portions using OpenCV-Python and CNN.

1) 입력 이미지 Gray Scale 적용 (OK)
    -
2) Gradient 추출 (추후)
3) Adaptive Threshold 적용한 image 반환, params 로 조정 값 받을 수 있도록 (OK)
4) Close 적용한 image 반환 (OK)
5) Long Line remove 적용한 image 반환 (추후)
6) 위 단계를 모두 거친 image 로부터 Contour 추출해서 Contours 반환
    - contours 사이즈 일정 크기 이상만 추철해서 반환하기
7) Contours 입력받아서 사각형(Rectangle)으로 원본이미지에 그리기

++++++++ 그 다음 생각하자 +++++++
8) Rectangle 표시된 부분 자르기
9) CNN 통해서 판별하기. 해당 Rectangle 이 text 인지 아닌지

"""
__author__ = "Woongje Han (niewoong)"
import cv2
import numpy as np
from matplotlib import pyplot as plt


PATH_SAMPLE_DIRECTORY = "C:/Users/viva/PycharmProjects/py-text-gotcha/images/"


def open_original(file_path):
    """ image file 을 읽어들여서 image 객체를 반환합니다.
    file_path 는 절대경로를 입력해야 합니다. todo (추후 확인 필요)
    """
    image_origin = cv2.imread(file_path)
    return image_origin


def get_gray(image_origin):
    """ image 객체를 받아서 Gray-scale 을 적용한 이미지 객체를 반환합니다.
    """
    image_copy_grey = cv2.cvtColor(image_origin, cv2.COLOR_BGR2GRAY)  # grey scale 로 복사합니다.
    return image_copy_grey


def get_adaptive_gaussian_threshold(image_gray, block_size=15, subtract_val=2):
    """ Gray-scale 이 적용된 이미지를 입력받아서 Adaptive Threshold 를 적용한 흑백(Binary) 이미지객체를 반환합니다.
    @Params
    * image_gray : Gray-scale 이 적용된 이미지객체
    * block_size : 픽셀에 적용할 threshold 값을 계산하기 위한 블럭 크기(홀수). 적용될 픽셀이 블럭의 중심이 됨.
    * subtract_val : 보정 상수
    """
    image_adaptive_gaussian = cv2.adaptiveThreshold(image_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                    cv2.THRESH_BINARY, block_size, subtract_val)
    return image_adaptive_gaussian  # todo 함수 안에서 생성한 객체 리턴하면 어떻게 되는거지? 주소값이 리턴되는 건가?? 소멸되지 않고?


def get_adaptive_mean_threshold(image_gray, block_size=15, subtract_val=2):
    """ Gray-scale 이 적용된 이미지를 입력받아서 Adaptive Threshold 를 적용한 흑백(Binary) 이미지객체를 반환합니다.
    @Params
    * image_gray : Gray-scale 이 적용된 이미지객체
    * block_size : 픽셀에 적용할   threshold 값을 계산하기 위한 블럭 크기(홀수). 적용될 픽셀이 블럭의 중심이 됨.
    * subtract_val : 보정 상수
    """
    image_adaptive_mean = cv2.adaptiveThreshold(image_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,
                                                block_size, subtract_val)
    return image_adaptive_mean


def get_closing(image_gray, kernel_size_row=3, kernel_size_col=3):
    """ Gray scale 이 적용된 이미지에 Morph Close 를 적용합니다.
    Closing : dilation 수행을 한 후 바로 erosion 수행을 하여 본래 이미지 크기로
    커널은 Image Transformation 을 결정하는 구조화된 요소
    커널의 크기가 크거나, 반복횟수가 많아지면 과하게 적용되어 경계가 없어질 수도 있다
    """
    # make kernel matrix for dilation and erosion (Use Numpy)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size_row, kernel_size_col))
    # closing 적용
    image_close = cv2.morphologyEx(image_gray, cv2.MORPH_CLOSE, kernel)
    return image_close


def get_contours(image_threshold):
    """ Threshold 가 적용된 이미지에 대하여 contour 를 추출하여 dictionary 형태로 반환합니다.
    todo cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE 옵션도 인자로 받고 default 로 두기
    """
    # contours는 point의 list형태.
    # hierarchy는 contours line의 계층 구조
    # Threshold 적용한 이미지에서 contour 들을 찾아서 contours 변수에 저장하기
    _, contours, _ = cv2.findContours(image_threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    return contours


def show_window(image, title):
    """ 윈도우를 열어서 이미지를 보여줍니다.
    """
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)  # 사용자가 크기 조절할 수 있는 윈도우 생성
    cv2.imshow(title, image)
    cv2.waitKey(0)  # 키보드 입력될 때 까지 계속 기다림
    cv2.destroyAllWindows()  # 이미지 윈도우 닫기


def save_image(image, file_path):
    """ 이미지를 file 로 저장합니다.
    """
    cv2.imwrite(file_path, image)


def main():
    file_path = PATH_SAMPLE_DIRECTORY + "ad_text2.jpg"
    image = open_original(file_path)
    image_gray = get_gray(image)
    image_threshold = get_adaptive_gaussian_threshold(image_gray)
    image_close = get_closing(image_threshold, 3, 3)
    show_window(image_threshold, "origin")
    show_window(image_close, "origin")

    return None


if __name__ == "__main__":
    main()
