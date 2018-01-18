#!/usr/bin/python
# -*- coding: utf-8 -*-
""" pre_precess.py 에서 정의된 이미지 처리(Image precessing)의
각 단계 및 최종 결과물에 대하여 테스트하고 분석할 수 있습니다.

    * 윈도우를 띄어서 진행단계의 이미지를 확인할 수 있습니다.
    * 이미지들을 양옆 및 위아래로 병합해여 비교할 수 있습니다.
"""

__author__ = "Woongje Han (niewoong)"
import pre_process as pp
import cv2
import numpy as np


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
    cv2.waitKey(0)  # wait until keyboard input
    cv2.destroyAllWindows()


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


def get_progress_image(resource_dir, filename_prefix):
    resource = resource_dir + filename_prefix + ".jpg"  # complete resource image path
    image_origin = pp.open_original(resource)
    image_origin = cv2.pyrUp(image_origin)  # size up ( x4 )
    comparing_images = []

    # Grey-Scale
    image_gray = pp.get_gray(image_origin)
    contours = pp.get_contours(image_gray)
    image_with_contours = pp.draw_contour_rect(image_origin, contours)

    compare_set = merge_vertical(image_gray, image_with_contours)
    comparing_images.append(compare_set)

    # Morph Gradient
    image_gradient = pp.get_gradient(image_gray)
    contours = pp.get_contours(image_gradient)
    image_with_contours = pp.draw_contour_rect(image_origin, contours)

    compare_set = merge_vertical(image_gradient, image_with_contours)
    comparing_images.append(compare_set)

    # Long line remove
    image_line_removed = pp.remove_long_line(image_gradient)
    contours = pp.get_contours(image_line_removed)
    image_with_contours = pp.draw_contour_rect(image_origin, contours)

    compare_set = merge_vertical(image_line_removed, image_with_contours)
    comparing_images.append(compare_set)

    # Threshold
    image_threshold = pp.get_threshold(image_line_removed)
    contours = pp.get_contours(image_threshold)
    image_with_contours = pp.draw_contour_rect(image_origin, contours)

    compare_set = merge_vertical(image_threshold, image_with_contours)
    comparing_images.append(compare_set)

    # Morph Close
    image_close = pp.get_closing(image_threshold)
    contours = pp.get_contours(image_close)
    image_with_contours = pp.draw_contour_rect(image_origin, contours)

    compare_set = merge_vertical(image_close, image_with_contours)
    comparing_images.append(compare_set)

    # Merge all step's images
    image_merged_all = np.hstack(comparing_images)
    # show_window(image_merged_all, 'image_merged_all')  # show all step
    # save_image(image_merged_all, filename_prefix)  # save all step image as a file
    # # save final result
    # save_image(image_with_contours, filename_prefix + '_final_')
    return pp.get_cropped_images(image_origin, contours)


def execute_test_set():
    # for i in range(1, 17):  # min <= i < max
    for i in (4, 8, 10, 13, 14):  # min <= i < max
        filename_prefix = "test (" + str(i) + ")"
        print(filename_prefix)
        crop_images = get_progress_image('images/', filename_prefix)
        count = 0
        f = open("results/" + filename_prefix + "_log3.txt", 'w')
        for crop_image in crop_images:
            count += 1
            # save_image(crop_image, filename_prefix + "crop_" + str(count))
            pp.image_to_text_file(crop_image, filename_prefix + "crop_" + str(count), f)
        f.close()


def main():
    execute_test_set()


if __name__ == "__main__":
    main()
