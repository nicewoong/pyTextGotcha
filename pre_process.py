#!/usr/bin/python
# -*- coding: utf-8 -*-
"""  Detect text in the image.
And processes the image to extract the text portions using OpenCV-Python and CNN.
"""


__author__ = "Woongje Han (niewoong)"
import cv2
import numpy as np
import yaml
import datetime
import pytesseract as ocr
from PIL import Image

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


def resize(image, flag=-1):
    """
    :param image:
    :param flag: flag > 0 이면 사이즈를 증가, flag < 0 (default)이면 사이즈를 축소한다.
    :return:
    """
    global configs
    standard_height = configs['resize_origin']['standard_height']
    standard_width = configs['resize_origin']['standard_width']
    # get image size
    height, width = image.shape[:2]
    # print original size
    print("width : " + str(width) + ", height : " + str(height))
    if (flag > 0 and height < standard_height) or (flag < 0 and height > standard_height):
        rate = standard_height / height
        w = round(width * rate)  # should be integer
        h = round(height * rate)  # should be integer
        image = cv2.resize(image, (w, h))
        print("after resize() (width : " + str(w) + ", height : " + str(h) + ")")
    elif (flag > 0 and width < standard_width) or (flag < 0 and height > standard_height):
        rate = standard_width / width
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
    copy = image_origin.copy()
    image_copy_grey = cv2.cvtColor(copy, cv2.COLOR_BGR2GRAY)  # grey scale 로 복사합니다.
    return image_copy_grey


def get_gradient(image_gray):
    copy = image_gray.copy()
    # get configs
    global configs
    kernel_size_row = configs['gradient']['kernel_size_row']
    kernel_size_col = configs['gradient']['kernel_size_col']
    # make kernel matrix for dilation and erosion
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size_row, kernel_size_col))
    # morph gradient
    image_gradient = cv2.morphologyEx(copy, cv2.MORPH_GRADIENT, kernel)
    return image_gradient


def get_threshold(image_gray):
    """ Gray-scale 이 적용된 이미지를 입력받아서 Adaptive Threshold 를 적용한 흑백(Binary) 이미지객체를 반환합니다.
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
    elif mode == 'global':
        image_threshold = get_otsu_threshold(image_gray)

    return image_threshold


def get_global_threshold(image_gray):
    ret, binary_image = cv2.threshold(image_gray, 180, 255, cv2.THRESH_BINARY)
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
    """
    global configs
    retrieve_mode = configs['contour']['retrieve_mode']  # integer
    approx_method = configs['contour']['approx_method']  # integer
    _, contours, _ = cv2.findContours(image_threshold, retrieve_mode, approx_method)
    return contours


def remove_long_line(image_binary, origin):
    # todo Consider removing the vertical line mainly. Horizontal lines should only remove large proportions in image
    copy = image_binary.copy()
    copy_rgb = origin.copy()
    # get configs
    global configs
    threshold = configs['remove_line']['threshold']
    min_line_length = configs['remove_line']['min_line_length']
    max_line_gap = configs['remove_line']['max_line_gap']

    # height, width = copy.shape[:2]  # get image size
    # min_line_length = height * 0.9
    # print(min_line_length)

    # find lines
    lines = cv2.HoughLinesP(copy, 1, np.pi / 180, threshold, min_line_length, max_line_gap)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]  # get end point of line : ( (x1, y1) , (x2, y2) )
            if x1 == x2 or y1 == y2:
                # remove line drawing black line
                cv2.line(copy, (x1, y1), (x2, y2), (0, 0, 0), 10)

    # show_window(copy)
    return copy


def draw_contour_rect(image_origin, contours, image_threshold):
    """ 이미지위에 찾은 Contours 를 기반으로 외각사각형을 그리고 해당 이미지를 반환합니다.
    """
    # get configs
    global configs
    min_width = configs['contour']['min_width']
    min_height = configs['contour']['min_height']
    rgb_copy = image_origin.copy()
    # Draw bounding rectangles
    for contour in contours:
        x, y, width, height = cv2.boundingRect(contour)  # 좌상단 꼭지점 좌표 , width, height
        # Rect 의 size 가 기준 이상인 것만 이미지 위에 그리기
        if width > min_width and height > min_height:
            cv2.rectangle(rgb_copy, (x, y), (x + width, y + height), (0, 255, 0), 2)  # 원본 이미지 위에 사각형 그리기!

    # cv2.drawContours(image, contours, -1, (0, 255, 0), 3)
    return rgb_copy


def get_crop_images(image_origin, contours):
    # get configs
    global configs
    min_width = configs['contour']['min_width']
    min_height = configs['contour']['min_height']
    margin = 7
    image_copy = image_origin.copy()
    origin_height, origin_width = image_copy.shape[:2]  # get image size
    crop_images = [image_copy]  # 자른 이미지를 하나씩 추가해서 저장할 리스트
    # todo contour 에서 margin 을 줬을 때 이미지 원본 영역을 벗어나지 않는지 체크해야한다.
    # todo 이미지 영역을 벗어날 경우 아래와 같은 에러 발생
    # todo ValueError: tile cannot extend outside image
    for contour in contours:
        x, y, width, height = cv2.boundingRect(contour)  # 좌상단 꼭지점 좌표 , width, height
        # Rect 의 size 가 기준 이상인 것만 담는다
        if width > min_width and height > min_height:
            crop_row_1 = (y - margin) if (y - margin) > 0 else y
            crop_row_2 = y + height + margin if (y + height + margin < origin_height) else height
            crop_col_1 = (x - margin) if (x - margin) > 0 else x
            crop_col_2 = x + width + margin if (x + width + margin < origin_width) else width
            # 행렬은 row col 순서!!! 햇갈리지 말자!
            crop = image_copy[crop_row_1: crop_row_2, crop_col_1: crop_col_2]  # trim한 결과를 img_trim에 담는
            crop_images.append(crop)
    return crop_images


def orc_test(image, title, f_stream=None):
    img = Image.fromarray(image)
    text = ocr.image_to_string(img, lang='kor')
    if f_stream is not None:
        f_stream.write("================ " + title + " ================ \n")
        f_stream.write(text + "\n")
    else:
        print("================ OCR result : " + title + "================")
        print(text)


def show_window(image, title='untitled'):
    """ 윈도우를 열어서 이미지를 보여줍니다.
    """
    height, width = image.shape[:2]  # get image size
    if height > 700:
        rate = 700 / height
        height = round(height * rate)
        width = round(width * rate)

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
    file_path = "results/" + name_prefix + "_" + current_datetime + ".jpg"
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
    image_origin = cv2.pyrUp(image_origin)  # size up ( x4 )
    comparing_images = []

    # Grey-Scale
    image_gray = get_gray(image_origin)
    contours = get_contours(image_gray)
    image_with_contours = draw_contour_rect(image_origin, contours, image_gray)

    compare_set = merge_vertical(image_gray, image_with_contours)
    comparing_images.append(compare_set)

    # Morph Gradient
    image_gradient = get_gradient(image_gray)
    contours = get_contours(image_gradient)
    image_with_contours = draw_contour_rect(image_origin, contours, image_gradient)

    compare_set = merge_vertical(image_gradient, image_with_contours)
    comparing_images.append(compare_set)

    # Long line remove
    image_line_removed = remove_long_line(image_gradient, image_origin)
    contours = get_contours(image_line_removed)
    image_with_contours = draw_contour_rect(image_origin, contours, image_line_removed)

    compare_set = merge_vertical(image_line_removed, image_with_contours)
    comparing_images.append(compare_set)

    # Threshold
    image_threshold = get_threshold(image_line_removed)
    contours = get_contours(image_threshold)
    image_with_contours = draw_contour_rect(image_origin, contours, image_threshold)

    compare_set = merge_vertical(image_threshold, image_with_contours)
    comparing_images.append(compare_set)

    # Morph Close
    image_close = get_closing(image_threshold)
    contours = get_contours(image_close)
    image_with_contours = draw_contour_rect(image_origin, contours, image_close)

    compare_set = merge_vertical(image_close, image_with_contours)
    comparing_images.append(compare_set)

    image_merged_all = np.hstack(comparing_images)
    # show_window(image_merged_all, 'image_merged_all')  # show all step
    # save_image(image_merged_all, filename_prefix)  # save all step image as a file

    # save final result
    save_image(image_with_contours, 'final_' + filename_prefix)
    return get_crop_images(image_origin, contours)


def execute_test_set():
    for i in range(1, 2):  # min <= i < max
        filename_prefix = "test (" + str(i) + ")"
        print(filename_prefix)
        crop_images = process_image('images/', filename_prefix, ".jpg")
        count = 0
        f = open("results/" + filename_prefix + "_log.txt", 'w')
        for crop_image in crop_images:
            count += 1
            save_image(crop_image, filename_prefix + "crop_" + str(count))
            orc_test(crop_image, filename_prefix + "crop_" + str(count), f)
        f.close()


def main():
    read_configs('config.yml')
    print_configs()
    execute_test_set()


if __name__ == "__main__":
    main()
