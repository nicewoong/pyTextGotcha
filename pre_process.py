#!/usr/bin/python
# -*- coding: utf-8 -*-
"""  Detect text in the image.
And processes the image to extract the text portions using OpenCV-Python and CNN.

1) Gray Scale 적용
2) Gradient 추출
3) Adaptive Threshold 적용
4) Close 적용
5) Long Line remove 적용
6) 위 단계를 모두 거친 image 로부터 Contour 추출
7) Contours 들 중 최소 사이즈 이상인 것들만 사각형(Rectangle)으로 원본이미지에 그리기

(본문에서)
저같은 경우,

gradient시 Kernel size는2 x 2
close시 Kernel size는 9 x 5
threshold시 block size는 3을 사용했습니다.



++++++++ 그 다음 생각하자 +++++++
8) Rectangle 표시된 부분 자르기
9) CNN 통해서 판별하기. 해당 Rectangle 이 text 인지 아닌지

"""
__author__ = "Woongje Han (niewoong)"
import cv2
import numpy as np
import yaml
from matplotlib import pyplot as plt


# configurations
configs = None


def read_configs(config_file):
    with open(config_file, 'r') as yml_file:
        configurations = yaml.load(yml_file)

    global configs
    configs = configurations
    return configurations


def print_configs():
    global configs
    for section in configs:
        print(section + ":")
        print(configs[section])


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


def get_gradient(image_gray, kernel_size_row=2, kernel_size_col=2):
    # make kernel matrix for dilation and erosion
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size_row, kernel_size_col))
    image_gradient = cv2.morphologyEx(image_gray, cv2.MORPH_GRADIENT, kernel)
    return image_gradient


def get_adaptive_gaussian_threshold(image_gray, block_size=30, subtract_val=30):
    """ Gray-scale 이 적용된 이미지를 입력받아서 Adaptive Threshold 를 적용한 흑백(Binary) 이미지객체를 반환합니다.

    :param image_gray: Gray-scale 이 적용된 이미지객체
    :param block_size: 픽셀에 적용할 threshold 값을 계산하기 위한 블럭 크기(홀수). 적용될 픽셀이 블럭의 중심이 됨.
    :param subtract_val: 보정 상수
    :return: Adaptive Threshold 를 적용한 흑백(Binary) 이미지객체
    """
    # image_adaptive_gaussian = cv2.adaptiveThreshold(image_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size, subtract_val)
    image_adaptive_gaussian = cv2.adaptiveThreshold(image_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, block_size, subtract_val)
    return image_adaptive_gaussian


def get_adaptive_mean_threshold(image_gray, block_size=3, subtract_val=12):
    """ Gray-scale 이 적용된 이미지를 입력받아서 Adaptive Threshold 를 적용한 흑백(Binary) 이미지객체를 반환합니다.
    @Params
    * image_gray : Gray-scale 이 적용된 이미지객체
    * block_size : 픽셀에 적용할   threshold 값을 계산하기 위한 블럭 크기(홀수). 적용될 픽셀이 블럭의 중심이 됨.
    * subtract_val : 보정 상수
    """
    # image_adaptive_mean = cv2.adaptiveThreshold(image_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,block_size, subtract_val)
    image_adaptive_mean = cv2.adaptiveThreshold(image_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, block_size, subtract_val)
    return image_adaptive_mean


def get_global_threshold(image_gray):
    ret, binary_image = cv2.threshold(image_gray, 127, 255, cv2.THRESH_BINARY)
    return binary_image


def get_otsu_threshold(image_gray):
    blur = cv2.GaussianBlur(image_gray, (5, 5), 0)  # Gaussian blur 를 통해 noise 를 제거한 후
    # global threshold with otsu's binarization
    ret3, image_otsu = cv2.threshold(image_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return image_otsu


def get_closing(image_gray, kernel_size_row=9, kernel_size_col=5):
    """ Gray scale 이 적용된 이미지에 Morph Close 를 적용합니다.
    Closing : dilation 수행을 한 후 바로 erosion 수행을 하여 본래 이미지 크기로
    커널은 Image Transformation 을 결정하는 구조화된 요소
    커널의 크기가 크거나, 반복횟수가 많아지면 과하게 적용되어 경계가 없어질 수도 있다
    """
    # make kernel matrix for dilation and erosion
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
    # _, contours, _ = cv2.findContours(image_threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    # _, contours, _ = cv2.findContours(image_threshold, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    _, contours, _ = cv2.findContours(image_threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours


def remove_vertical_line(image, threshold=100, min_line_length=80, max_line_gap=5):
    lines = cv2.HoughLinesP(image, 1, np.pi / 180, threshold, min_line_length, max_line_gap)
    for line in lines:
        x1, y1, x2, y2 = line[0]  # get line end point ( (x1, y1) , (x2, y2) )
        # remove line drawing black line
        cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return image


def draw_contour_rect(image, contours, min_width=3, min_height=3):
    """ 이미지위에 찾은 Contours 를 기반으로 외각사각형을 그리고 해당 이미지를 반환합니다.
    todo contour 의 색상이나 두깨도 인자로 받아서 설정할 수 있도록 변경하기
    """
    # 찾은 contours 를 입력받은 이미지위에 그리기
    # image_with_contours = cv2.drawContours(image, contours, -1, (0, 255, 0), 2)

    # Draw bounding rectangle
    for contour in contours:
        x, y, width, height = cv2.boundingRect(contour)  # 좌상단 꼭지점 좌표 , width, height
        # todo Rect 의 size 가 기준 이상인 것만 이미지 위에 그리기
        if width > min_width and height > min_height:
            cv2.rectangle(image, (x, y), (x+width, y+height), (0, 255, 0), 2)  # 원본 이미지 위에 사각형 그리기!

    return image


def show_window(image, title):
    """ 윈도우를 열어서 이미지를 보여줍니다.
    """
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)  # 사용자가 크기 조절할 수 있는 윈도우 생성
    cv2.resizeWindow(title, 300, 500)
    cv2.imshow(title, image)
    cv2.waitKey(0)  # 키보드 입력될 때 까지 계속 기다림
    cv2.destroyAllWindows()  # 이미지 윈도우 닫기


def save_image(image, file_path):
    """ 이미지를 file 로 저장합니다.
    """
    cv2.imwrite(file_path, image)


def process_image():
    """ 영향을 미치는 변수를 다양하게 적용해보면서 맞추어야 합니다.
    :return:
    """
    block_size = 9 # Threshold (Odd number !!)
    subtract_val = 12  # Threshold
    gradient_kernel_size_row = 2  # Gradient Kernel Size
    gradient_kernel_size_col = 2  # Gradient Kernel Size
    close_kernel_size_row = 2  # Closing Kernel Size
    close_kernel_size_col = 2  # Closing Kernel Size
    min_width = 4  # Minimum Contour Rectangle Size
    min_height = 10  # Minimum Contour Rectangle Size
    threshold = 100  # Long Line Remove Precision
    min_line_length = 100  # Long Line Remove  Minimum Line Length
    max_line_gap = 5  # Long Line Remove Maximum Line Gap

    # todo 위 경우 말고도 각각 메서드마다 적용되는 cv2.ABCDE 형식의 상수값도 모두 조정해줄 수 있어야한다.

    file_path = "images/car.png"
    resulst_file_name = "block_s-" + str(block_size) + "subt-" + str(subtract_val) + ".png"
    image = open_original(file_path)
    # image = cv2.resize(image, (512, 712))

    # Grey-Scale
    image_gray = get_gray(image)
    show_window(image_gray, 'image_gray')  # show
    save_image(image_gray, '_gray.png')  # save image as a file

    contours = get_contours(image_gray)
    image_with_contours = draw_contour_rect(image, contours, min_width, min_height)
    show_window(image_with_contours, "result")  # show

    # Morph Gradient
    image_gradient = get_gradient(image_gray, gradient_kernel_size_row, gradient_kernel_size_col)
    show_window(image_gradient, "image_gradient")  # show
    save_image(image_gradient, '_gradient.png')  # save image as a file

    contours = get_contours(image_gradient)
    image_with_contours = draw_contour_rect(image, contours, min_width, min_height)
    show_window(image_with_contours, "result")  # show

    # Threshold
    image_threshold = get_adaptive_mean_threshold(image_gradient, block_size, subtract_val)
    show_window(image_threshold, "adaptive_threshold")  # show
    save_image(image_threshold, '_threshold.png')  # save image as a file

    contours = get_contours(image_threshold)
    image_with_contours = draw_contour_rect(image, contours, min_width, min_height)
    show_window(image_with_contours, "result")  # show

    # Morph Close
    image_close = get_closing(image_threshold, close_kernel_size_row, close_kernel_size_col)
    show_window(image_close, "image_close")  # show
    save_image(image_close, '_close.png')  # save image as a file

    contours = get_contours(image_close)
    image_with_contours = draw_contour_rect(image, contours, min_width, min_height)
    show_window(image_with_contours, "result")  # show

    # Long line remove
    remove_vertical_line(image_close, threshold, min_line_length, max_line_gap)
    show_window(image_close, "removed line")
    save_image(image_close, '_remove_line.png')  # save image as a file

    # Contours
    contours = get_contours(image_close)
    image_with_contours = draw_contour_rect(image, contours, min_width, min_height)
    show_window(image_with_contours, "result")  # show
    save_image(image_with_contours, '_contour.png')  # save image as a file

    # Gray
    # Gradient
    # Adaptive Threshold  (Binary Invert)
    # Close
    # Contours

    return None


def main():
    read_configs('config.yml')
    print_configs()


if __name__ == "__main__":
    main()
