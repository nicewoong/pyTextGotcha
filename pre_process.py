#!/usr/bin/python
# -*- coding: utf-8 -*-
"""  Detect text in the image.
And processes the image to extract the text portions using OpenCV-Python and CNN.
"""
__author__ = "Woongje Han (niewoong)"
import cv2
import numpy as np
import yaml
from matplotlib import pyplot as plt


# configurations to read from YAML file
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


def resize(image):
    # todo get config from yml file
    max_height = 400
    # get image size
    height, width = image.shape[:2]
    # print original size
    print("width : " + str(width) + ", height : " + str(height))
    # resize if too large
    if height > max_height:
        rate = max_height / height
        w = round(width * rate)  # should be integer
        h = round(height * rate)  # should be integer
        image = cv2.resize(image, (w, h))
        print("width : " + str(w) + ", height : " + str(h))
    return image


def open_original(file_path):
    """ image file 을 읽어들여서 image 객체를 반환합니다.
    """
    image_origin = cv2.imread(file_path)
    return image_origin


def get_gray(image_origin):
    """ image 객체를 받아서 Gray-scale 을 적용한 이미지 객체를 반환합니다.
    """
    image_copy_grey = cv2.cvtColor(image_origin, cv2.COLOR_BGR2GRAY)  # grey scale 로 복사합니다.
    return image_copy_grey


def get_gradient(image_gray):
    # get configs
    global configs
    kernel_size_row = configs['gradient']['kernel_size_row']
    kernel_size_col = configs['gradient']['kernel_size_col']
    # make kernel matrix for dilation and erosion
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size_row, kernel_size_col))
    # morph gradient
    image_gradient = cv2.morphologyEx(image_gray, cv2.MORPH_GRADIENT, kernel)
    return image_gradient


def get_adaptive_gaussian_threshold(image_gray):
    """ Gray-scale 이 적용된 이미지를 입력받아서 Adaptive Threshold 를 적용한 흑백(Binary) 이미지객체를 반환합니다.

    :param image_gray: Gray-scale 이 적용된 이미지객체
    :return: Adaptive Threshold 를 적용한 흑백(Binary) 이미지객체
    """
    # get configs
    global configs
    block_size = configs['threshold']['block_size']
    subtract_val = configs['threshold']['subtract_val']
    # adaptive threshold
    image_adaptive_gaussian = cv2.adaptiveThreshold(image_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, block_size, subtract_val)
    return image_adaptive_gaussian


def get_adaptive_mean_threshold(image_gray):
    """ Gray-scale 이 적용된 이미지를 입력받아서 Adaptive Threshold 를 적용한 흑백(Binary) 이미지객체를 반환합니다.
    todo cv2.adaptiveThreshold() 의 세 번째 인자를 configuration file 을 통해 설정할 수 있도록 하고 메서드를 하나로 통일하기
    """
    # get configs
    global configs
    block_size = configs['threshold']['block_size']
    subtract_val = configs['threshold']['subtract_val']
    # adaptive threshold
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


def get_closing(image_gray):
    """ Gray scale 이 적용된 이미지에 Morph Close 를 적용합니다.
    Closing : dilation 수행을 한 후 바로 erosion 수행을 하여 본래 이미지 크기로
    커널은 Image Transformation 을 결정하는 구조화된 요소
    커널의 크기가 크거나, 반복횟수가 많아지면 과하게 적용되어 경계가 없어질 수도 있다
    """
    # get configs
    global configs
    kernel_size_row = configs['close']['kernel_size_row']
    kernel_size_col = configs['close']['kernel_size_col']
    # make kernel matrix for dilation and erosion
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size_row, kernel_size_col))
    # close
    image_close = cv2.morphologyEx(image_gray, cv2.MORPH_CLOSE, kernel)
    return image_close


def get_contours(image_threshold):
    """ Threshold 가 적용된 이미지에 대하여 contour 를 추출하여 dictionary 형태로 반환합니다.
    todo cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE 옵션도 configuration 에서 설정할 수 있도록 하라
    """
    # contours는 point의 list형태.
    # hierarchy는 contours line의 계층 구조
    # Threshold 적용한 이미지에서 contour 들을 찾아서 contours 변수에 저장하기
    # _, contours, _ = cv2.findContours(image_threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    # _, contours, _ = cv2.findContours(image_threshold, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    _, contours, _ = cv2.findContours(image_threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours


def remove_vertical_line(image):
    # get configs
    global configs
    threshold = configs['remove_line']['threshold']
    min_line_length = configs['remove_line']['min_line_length']
    max_line_gap = configs['remove_line']['max_line_gap']
    # find lines
    lines = cv2.HoughLinesP(image, 1, np.pi / 180, threshold, min_line_length, max_line_gap)
    for line in lines:
        x1, y1, x2, y2 = line[0]  # get end point of line : ( (x1, y1) , (x2, y2) )
        # remove line drawing black line
        cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return image


def draw_contour_rect(image, contours):
    """ 이미지위에 찾은 Contours 를 기반으로 외각사각형을 그리고 해당 이미지를 반환합니다.
    todo contour 의 색상이나 두깨도 인자로 받아서 설정할 수 있도록 변경하기
    """
    # get configs
    global configs
    min_width = configs['contour']['min_width']
    min_height = configs['contour']['min_height']

    # Draw bounding rectangles
    for contour in contours:
        x, y, width, height = cv2.boundingRect(contour)  # 좌상단 꼭지점 좌표 , width, height
        # todo Rect 의 size 가 기준 이상인 것만 이미지 위에 그리기
        if width > min_width and height > min_height:
            cv2.rectangle(image, (x, y), (x+width, y+height), (0, 255, 0), 2)  # 원본 이미지 위에 사각형 그리기!

    return image


def show_window(image, title):
    """ 윈도우를 열어서 이미지를 보여줍니다.
    """
    height, width = image.shape[:2]  # get image size
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)  # Create a window that the user can resize
    cv2.resizeWindow(title, width, height)  # resize window according to the size of the image
    cv2.imshow(title, image)  # open image window
    cv2.waitKey(0)  # Continue to wait until keyboard input
    cv2.destroyAllWindows()


def save_image(image, file_path):
    """ 이미지를 file 로 저장합니다.
    """
    cv2.imwrite(file_path, image)


def process_image():
    """ 영향을 미치는 변수를 다양하게 적용해보면서 맞추어야 합니다.
    :return:
    """
    file_path = "images/car.png"
    image = open_original(file_path)

    # Grey-Scale
    image_gray = get_gray(image)
    show_window(image_gray, 'image_gray')  # show
    save_image(image_gray, '_gray.png')  # save image as a file

    contours = get_contours(image_gray)
    image_with_contours = draw_contour_rect(image, contours)
    show_window(image_with_contours, "result")  # show

    # Morph Gradient
    image_gradient = get_gradient(image_gray)
    show_window(image_gradient, "image_gradient")  # show
    save_image(image_gradient, '_gradient.png')  # save image as a file

    contours = get_contours(image_gradient)
    image_with_contours = draw_contour_rect(image, contours)
    show_window(image_with_contours, "result")  # show

    # Threshold
    image_threshold = get_adaptive_mean_threshold(image_gradient)
    show_window(image_threshold, "adaptive_threshold")  # show
    save_image(image_threshold, '_threshold.png')  # save image as a file

    contours = get_contours(image_threshold)
    image_with_contours = draw_contour_rect(image, contours)
    show_window(image_with_contours, "result")  # show

    # Morph Close
    image_close = get_closing(image_threshold)
    show_window(image_close, "image_close")  # show
    save_image(image_close, '_close.png')  # save image as a file

    contours = get_contours(image_close)
    image_with_contours = draw_contour_rect(image, contours)
    show_window(image_with_contours, "result")  # show

    # Long line remove
    remove_vertical_line(image_close)
    show_window(image_close, "removed line")
    save_image(image_close, '_remove_line.png')  # save image as a file

    # Contours
    contours = get_contours(image_close)
    image_with_contours = draw_contour_rect(image, contours)
    show_window(image_with_contours, "result")  # show
    save_image(image_with_contours, '_contour.png')  # save image as a file

    return None


def test_resize():
    file_path = "images/car.png"
    image = open_original(file_path)
    image = resize(image)
    show_window(image, "result")  # show


def main():
    read_configs('config.yml')
    print_configs()
    test_resize()


if __name__ == "__main__":
    main()
