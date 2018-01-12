#!/usr/bin/python
# -*- coding: utf-8 -*-
"""  Detect text in the image.
And processes the image to extract the text portions using OpenCV-Python and CNN.
"""
import datetime

__author__ = "Woongje Han (niewoong)"
import cv2
import numpy as np
import yaml
import os

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
    # todo get max_height max_width from yml file and apply it to resource
    global configs
    max_height = configs['resize_origin']['max_height']
    max_width = configs['resize_origin']['max_width']
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
        print("after resize() (width : " + str(w) + ", height : " + str(h) + ")")
    elif width > max_width:
        rate = max_width / width
        w = round(width * rate)  # should be integer
        h = round(height * rate)  # should be integer
        image = cv2.resize(image, (w, h))
        print("after resize() (width : " + str(w) + ", height : " + str(h) + ")")
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


def get_threshold(image_gray):
    """ Gray-scale 이 적용된 이미지를 입력받아서 Adaptive Threshold 를 적용한 흑백(Binary) 이미지객체를 반환합니다.
    todo cv2.adaptiveThreshold() 의 세 번째 인자를 configuration file 을 통해 설정할 수 있도록 하고 메서드를 하나로 통일하기
    """
    # get configs
    global configs
    mode = configs['threshold']['mode']
    block_size = configs['threshold']['block_size']
    subtract_val = configs['threshold']['subtract_val']

    if mode == 'mean':
        # adaptive threshold - mean
        image_threshold = cv2.adaptiveThreshold(image_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                                        cv2.THRESH_BINARY_INV, block_size, subtract_val)
    elif mode == 'gaussian':
        image_threshold = cv2.adaptiveThreshold(image_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                        cv2.THRESH_BINARY_INV, block_size, subtract_val)

    return image_threshold


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
    """ Threshold 가 적용된 이미지에 대하여 contour 리스트를 추출하여 dictionary 형태로 반환합니다.
    todo Consider that retrieve_mode and approx_method can be changed by constant values ​​defined in cv2
    """
    global configs
    retrieve_mode = configs['contour']['retrieve_mode']  # integer
    approx_method = configs['contour']['approx_method']  # integer
    _, contours, _ = cv2.findContours(image_threshold, retrieve_mode, approx_method)
    return contours


def remove_vertical_line(image):
    # todo Consider removing the vertical line mainly. Horizontal lines should only remove large proportions in image
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
        # Rect 의 size 가 기준 이상인 것만 이미지 위에 그리기
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


def save_image(image, name_prefix):
    """ 이미지를 file 로 저장합니다.
    """
    d_date = datetime.datetime.now()
    current_datetime = d_date.strftime("%Y%m%d%I%M%S")
    file_path = "results/" + name_prefix + "_" + current_datetime + ".png"
    cv2.imwrite(file_path, image)


def merge_horizontal(image_gray, image_contours):
    # Make the grey scale image have three channels
    image_cr = cv2.cvtColor(image_gray, cv2.COLOR_GRAY2BGR)
    # Merge image horizontally
    numpy_horizontal = np.hstack((image_cr, image_contours))
    # numpy_horizontal_concat = np.concatenate((image, image_contours), axis=1)
    return numpy_horizontal


def merge_vertical(image_gray, image_contours):
    # Make the grey scale image have three channels
    image_cr = cv2.cvtColor(image_gray, cv2.COLOR_GRAY2BGR)
    # Merge image horizontally
    numpy_vertical = np.vstack((image_cr, image_contours))
    return numpy_vertical


def process_image(resource_dir, filename_prefix, extension):
    resource = resource_dir + filename_prefix + extension
    image_origin = open_original(resource)
    image_origin = resize(image_origin)
    comparing_images = []

    # Grey-Scale
    image_gray = get_gray(image_origin)
    contours = get_contours(image_gray)
    image_with_contours = draw_contour_rect(image_origin, contours)

    compare_set = merge_vertical(image_gray, image_with_contours)
    comparing_images.append(compare_set)
    # show_window(merge_horizontal(image_gray, image_with_contours), 'image_gray')  # show

    # Morph Gradient
    image_gradient = get_gradient(image_gray)
    contours = get_contours(image_gradient)
    image_with_contours = draw_contour_rect(image_origin, contours)

    compare_set = merge_vertical(image_gradient, image_with_contours)
    comparing_images.append(compare_set)
    # show_window(merge_horizontal(image_gradient, image_with_contours), 'image_gradient')  # show

    # Threshold
    image_threshold = get_threshold(image_gradient)
    contours = get_contours(image_threshold)
    image_with_contours = draw_contour_rect(image_origin, contours)

    compare_set = merge_vertical(image_threshold, image_with_contours)
    comparing_images.append(compare_set)
    # show_window(merge_horizontal(image_threshold, image_with_contours), 'image_threshold')  # show

    # Morph Close
    image_close = get_closing(image_threshold)
    contours = get_contours(image_close)
    image_with_contours = draw_contour_rect(image_origin, contours)

    compare_set = merge_vertical(image_close, image_with_contours)
    comparing_images.append(compare_set)
    # show_window(merge_horizontal(image_close, image_with_contours), 'image_close')  # show

    # Long line remove
    remove_vertical_line(image_close)
    contours = get_contours(image_close)
    image_with_contours = draw_contour_rect(image_origin, contours)

    compare_set = merge_vertical(image_close, image_with_contours)
    comparing_images.append(compare_set)
    # show_window(merge_horizontal(image_close, image_with_contours), 'remove_vertical_line')  # show

    image_merged_all = np.hstack(comparing_images)
    show_window(image_merged_all, 'image_merged_all')  # show

    save_image(image_merged_all, filename_prefix)  # save image as a file


def execute_test_set():
    for i in range(1, 19):  # 1 <= i < 20
        filename_prefix = "test_" + str(i)
        print(filename_prefix    )
        process_image('cut_resources/', filename_prefix, ".PNG")


def main():
    read_configs('config.yml')
    print_configs()
    execute_test_set()


if __name__ == "__main__":
    main()
