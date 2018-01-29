#!/usr/bin/python
# -*- coding: utf-8 -*-
""" pre_precess.py 에서 정의된 이미지 처리(Image precessing)의
각 단계 및 최종 결과물에 대하여 테스트하고 분석할 수 있습니다.

    * 윈도우를 띄어서 진행단계의 이미지를 확인할 수 있습니다.
    * 이미지들을 양옆 및 위아래로 병합해여 비교할 수 있습니다.
"""

__author__ = "Woongje Han (niewoong)"
import pre_process as pp
import judge_text as jt
import cv2
import numpy as np
import os


def show_window(image, title='untitled', max_height=700):
    """ 이미지 윈도우를 열어서 보여줍니다.

    :param image: 보여줄 이미지 (OpenCV image 객체)
    :param title: 윈도우 제목
    :param max_height: 이미지 윈도우 사이즈의 최대 높이
    :return:
    """
    height, width = image.shape[:2]  # get image size
    if height > max_height:  # adjust window size if too large
        rate = max_height / height
        height = round(height * rate)
        width = round(width * rate)  # apply the same rate to width

    cv2.namedWindow(title, cv2.WINDOW_NORMAL)  # Create a window that the user can resize
    cv2.resizeWindow(title, width, height)  # resize window according to the size of the image
    cv2.imshow(title, image)  # open image window
    key = cv2.waitKey(0)  # wait until keyboard input
    cv2.destroyAllWindows()
    return key


def merge_horizontal(image_gray, image_bgr):
    """ Height 사이즈가 같은 두 이미지를 옆으로(Horizontally) 병합 합니다.
    이미지 처리(Image processing) 단계를 원본과 비교하기위한 목적으로,
    2차원(2 dimension) 흑백 이미지와 3차원(3 dimension) BGR 컬리 이미지를 인자로 받아 병합합니다.

    :param image_gray: 2차원(2 dimension) 흑백 이미지
    :param image_bgr: 3차원(3 dimension) BGR 컬리 이미지
    :return: 옆으로(Horizontally) 병합된 이미지
    """
    # Make the grey scale image have 3 channels
    image_cr = cv2.cvtColor(image_gray, cv2.COLOR_GRAY2BGR)
    # Merge image horizontally
    numpy_horizontal = np.hstack((image_cr, image_bgr))
    # numpy_horizontal_concat = np.concatenate((image, image_contours), axis=1)
    return numpy_horizontal


def merge_vertical(image_gray, image_bgr):
    """ Width 사이즈가 같은 두 이미지를 위아래로(Vertically) 병합 합니다.
    이미지 처리(Image processing) 단계를 원본과 비교하기위한 목적으로,
    2차원(2 dimension) 흑백 이미지와 3차원(3 dimension) BGR 컬리 이미지를 인자로 받아 병합합니다.

    :param image_gray: 2차원(2 dimension) 흑백 이미지
    :param image_bgr: 3차원(3 dimension) BGR 컬리 이미지
    :return: 위아래로(Vertically) 병합된 이미지
    """
    # Make the grey scale image have 3 channels
    image_cr = cv2.cvtColor(image_gray, cv2.COLOR_GRAY2BGR)
    # Merge image horizontally
    numpy_vertical = np.vstack((image_cr, image_bgr))
    return numpy_vertical


def detect_line(image_binary):
    """ 이미지에서 직선을 찾아서 초록색으로 표시한 결과를 반환합니다.

    :param image_binary: 흑백(Binary) OpenCV image (2 dimension)
    :return: 라인이 삭제된 이미지 (OpenCV image)
    """
    copy = image_binary.copy()  # copy the image to be processed
    copy_rbg = cv2.cvtColor(copy, cv2.COLOR_GRAY2RGB)
    # get configs
    threshold = pp.configs['remove_line']['threshold']
    min_line_length = pp.configs['remove_line']['min_line_length']
    max_line_gap = pp.configs['remove_line']['max_line_gap']

    # fine and draw lines
    lines = cv2.HoughLinesP(copy, 1, np.pi / 180, threshold, np.array([]), min_line_length, max_line_gap)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]  # get end point of line : ( (x1, y1) , (x2, y2) )
            # slop = 0
            # if x2 != x1:
            #     slop = abs((y2-y1) / (x2-x1))
            # if slop < 0.5 or slop > 50 or x2 == x1:  # only vertical or parallel lines.
            #  remove line drawing black line
            cv2.line(copy_rbg, (x1, y1), (x2, y2), (0, 155, 0), 2)
    return copy_rbg


def get_step_compare_image(path_of_image):
    """ 이미지 프로세싱 전 단계의 중간 결과물을 하나로 병합하여 반환합니다.

    :param path_of_image:
    :return:
    """
    # open original image
    image_origin = pp.open_original(path_of_image)
    # size up ( x4 )
    image_origin = cv2.pyrUp(image_origin)
    comparing_images = []

    # Grey-Scale
    image_gray = pp.get_gray(image_origin)
    contours = pp.get_contours(image_gray)
    image_with_contours = pp.draw_contour_rect(image_origin, contours)
    # merge two image vertically
    compare_set = merge_vertical(image_gray, image_with_contours)
    comparing_images.append(compare_set)

    # Morph Gradient
    image_gradient = pp.get_gradient(image_gray)
    # image_gradient = pp.get_canny(image_gray)
    contours = pp.get_contours(image_gradient)
    image_with_contours = pp.draw_contour_rect(image_origin, contours)
    # merge two current step image vertically
    compare_set = merge_vertical(image_gradient, image_with_contours)
    comparing_images.append(compare_set)

    # Threshold
    image_threshold = pp.get_threshold(image_gradient)
    contours = pp.get_contours(image_threshold)
    image_with_contours = pp.draw_contour_rect(image_origin, contours)
    # merge two image vertically
    compare_set = merge_vertical(image_threshold, image_with_contours)
    comparing_images.append(compare_set)

    # Long line remove
    image_line_removed = pp.remove_long_line(image_threshold)
    contours = pp.get_contours(image_line_removed)
    image_with_contours = pp.draw_contour_rect(image_origin, contours)
    # merge two image vertically
    compare_set = merge_vertical(image_line_removed, image_with_contours)
    comparing_images.append(compare_set)

    # Morph Close
    image_close = pp.get_closing(image_line_removed)
    contours = pp.get_contours(image_close)
    image_with_contours = pp.draw_contour_rect(image_origin, contours)
    # merge two image vertically
    compare_set = merge_vertical(image_close, image_with_contours)
    comparing_images.append(compare_set)

    # Merge all step's images horizontally
    image_merged_all = np.hstack(comparing_images)

    return image_merged_all


def get_image_with_contours(path_of_image):
    """ 이미지 프로세싱을 거친 후,
    최종적으로 얻은 Contours 를 원본 이미지 위에 그려서 반환합니다.

    :param path_of_image:
    :return:
    """
    # open original image
    image_origin = pp.open_original(path_of_image)
    # size up the resource ( x4 )
    image_origin = cv2.pyrUp(image_origin)
    # Grey-Scale
    image_gray = pp.get_gray(image_origin)
    # Morph Gradient
    image_gradient = pp.get_gradient(image_gray)
    # Threshold
    image_threshold = pp.get_threshold(image_gradient)
    # Long line remove
    image_line_removed = pp.remove_long_line(image_threshold)
    # Morph Close
    image_close = pp.get_closing(image_line_removed)
    # Get contours and Draw it on the original image
    contours = pp.get_contours(image_close)
    image_with_contours = pp.draw_contour_rect(image_origin, contours)
    return image_with_contours


def get_file_list(path):
    """ path 가 가리키는 directory 의 모든 파일명을 읽어서 string 으로 반환합니다.
    파일명은 Absolute path 가 포함된 이름입니다.

    :param path: 읽어 들일 directory 의 절대경로
    :return: directory 의 모든 file path 을 String 형으로 Array 에 담아 반환
    """
    image_path_list = []
    for root, dirs, files in os.walk(path):
        root_path = os.path.join(os.path.abspath(path), root)
        for file in files:
            file_path = os.path.join(root_path, file)
            image_path_list.append(file_path)

    return image_path_list


def read_text_from_image(image_path):
    messages = []
    cropped_images = pp.process_image(image_path)
    count = 1
    for cropped in cropped_images:
        count += 1
        # gray_copy = pp.get_gray(cropped)
        # gradient_copy = pp.get_gradient(gray_copy)
        # gradient_copy = cv2.cvtColor(gradient_copy, cv2.COLOR_GRAY2BGR)
        # answer = jt.get_answer_from_cv2_Image(gradient_copy)
        # print(answer)
        msg = pp.get_text_from_image(cropped)
        messages.append(msg)

    return messages


def get_image_with_lines(image_path):
    image_origin = pp.open_original(image_path)
    image_origin = cv2.pyrUp(image_origin)
    # Grey-Scale
    image_gray = pp.get_gray(image_origin)
    # Morph Gradient
    image_gradient = pp.get_gradient(image_gray)
    # Threshold
    image_threshold = pp.get_threshold(image_gradient)
    # find and draw lines
    image_line_removed = detect_line(image_threshold)
    return image_line_removed


def main():
    pp.read_configs('images/config_screenshot.yml')  # set configs  todo parameter 에서 옵션값으로 입력받도록 바꾸기
    image_path = 'images/test (11).jpg'  # todo 실행시 parameter 로 image 를 입력받도록 바꾸기
    result = get_step_compare_image(image_path)
    # show result
    show_window(result, 'all steps')
    result = get_image_with_contours(image_path)
    show_window(result, 'final result')


if __name__ == "__main__":
    main()
