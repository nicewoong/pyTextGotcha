#!/usr/bin/python
# -*- coding: utf-8 -*-

""" 이미지에서 글자 부분을 추출하기 위해 이미지 처리(Image precessing)를 합니다.

* 이미지에서 글자로 추정되는 부분을 찾아 표시하거나 해당 영역을 잘라낼 수 있습니다.
* 글자로 추정되는 부분을 더 잘 찾기 위해 이미지 처리(Image precessing)를 거칩니다.
* 이미지 처리(Image precessing)는 5단계로 구성됩니다.
  process_image() 에서 순서를 변경하여 적용할 수 있습니다.
    1) Gray-scale 적용
    2) Morph Gradient 적용
    3) Long Line Removal 적용
    4) Threshold 적용
    5) Close 적용

* 위 이미지 처리(Image precessing) 단계를 거친 후 Contour(경계영역)를 추출하면
  글자로 추정되는 영역을 발견할 수 있습니다.

* 각 단계의 함수 내부에서 사용되는 상수 값들은 configs.yml 파일에서 설정가능합니다.

* 아래 import 목록에 해당하는 파이썬 패키지가 개발환경에 설치되어야 합니다.
  특히 OCR(문자 인식)을 위한 pytesseract 를 사용하기 위해서는
  각 언어팩을 함께 설치해야함에 주의하세요.
  이미지처리에는 주로 OpenCV 라이브러리를 활용합니다.


* See: https://github.com/nicewoong/pyTextGotcha
todo 사용한 오픈소스 라이센스 표시하기
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
    """ .yml file 을 읽어서 configuration 값의 객체를 갖습니다.

    :param config_file:
    :return: 읽은 configuration 을 담고있는 dictionary 형태로 반환
    """
    # read contents from .yam config file
    with open(config_file, 'r') as yml_file:
        configurations = yaml.load(yml_file)  # use 'yaml' package to read .yml file

    global configs  # global var : configs
    configs = configurations  # set configs
    return configurations  # return read configurations


def print_configs():
    """ 전역변수 configs 에 저장된 configuration 내용을 출력합니다.

    :return: None
    """
    global configs  # refer global variable : configs
    for section in configs:
        print(section + ":")
        print(configs[section])


def resize(image, flag=-1):
    """ Configuration 의 width, height 값을 기준으로 이미지 사이즈를 변경합니다.

    :param image - cv2 이미지 객체
    :param flag - flag > 0 이면 사이즈를 증가, flag < 0 (default)이면 사이즈를 축소
    :return: image_copy - 사이즈가 변환된 이미지
    """
    # get configs
    global configs
    standard_height = configs['resize_origin']['standard_height']
    standard_width = configs['resize_origin']['standard_width']
    # get image size
    height, width = image.shape[:2]
    image_copy = image.copy()
    # print original size (width, height)
    print("origin (width : " + str(width) + ", height : " + str(height) + ")")
    rate = 1  # default
    if (flag > 0 and height < standard_height) or (flag < 0 and height > standard_height):  # Resize based on height
        rate = standard_height / height
    elif (flag > 0 and width < standard_width) or (flag < 0 and height > standard_height):  # Resize based on width
        rate = standard_width / width
    # resize
    w = round(width * rate)  # should be integer
    h = round(height * rate)  # should be integer
    image_copy = cv2.resize(image_copy, (w, h))
    # print modified size (width, height)
    print("after resize : (width : " + str(w) + ", height : " + str(h) + ")")
    return image_copy


def open_original(file_path):
    """ image file 을 읽어들여서 OpenCV image 객체로 반환합니다.

    :param file_path:  경로를 포함한 이미지 파일
    :return:  OpenCV 의 BGR image 객체 (3 dimension)
    """
    image_origin = cv2.imread(file_path)  # read image from file
    return image_origin


def get_gray(image_origin):
    """ image 객체를 인자로 받아서 Gray-scale 을 적용한 2차원 이미지 객체로 반환합니다.
    이 때 인자로 입력되는 이미지는 BGR 컬러 이미지여야 합니다.

    :param image_origin: OpenCV 의 BGR image 객체 (3 dimension)
    :return: gray-scale 이 적용된 image 객체 (2 dimension)
    """
    copy = image_origin.copy()  # copy the image to be processed
    image_grey = cv2.cvtColor(copy, cv2.COLOR_BGR2GRAY)  # apply gray-scale to the image
    return image_grey


def get_gradient(image_gray):
    """ 이미지에 Dilation 과 Erosion 을 적용하여 그 차이를 이용해 윤곽선을 추출합니다.
    이 때 인자로 입력되는 이미지는 Gray scale 이 적용된 2차원 이미지여야 합니다.

    :param image_gray: Gray-scale 이 적용된 OpenCV image (2 dimension)
    :return: 윤곽선을 추출한 결과 이미지 (OpenCV image)
    """
    copy = image_gray.copy()  # copy the image to be processed
    # get configs
    global configs
    kernel_size_row = configs['gradient']['kernel_size_row']
    kernel_size_col = configs['gradient']['kernel_size_col']
    # make kernel matrix for dilation and erosion
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size_row, kernel_size_col))
    # morph gradient
    image_gradient = cv2.morphologyEx(copy, cv2.MORPH_GRADIENT, kernel)
    return image_gradient


def remove_long_line(image_binary):
    """ 이미지에서 직선을 찾아서 삭제합니다.
    글자 경계를 찾을 때 방해가 되는 직선을 찾아서 삭제합니다.
    이 때 인자로 입력되는 이미지 2 차원(2 dimension) 흑백(Binary) 이미지여야 합니다.
    직선을 삭제할 때는 해당 라인을 검정색으로 그려 덮어 씌웁니다. 

    :param image_binary: 흑백(Binary) OpenCV image (2 dimension)
    :return: 라인이 삭제된 이미지 (OpenCV image)
    """
    copy = image_binary.copy()  # copy the image to be processed
    # get configs
    global configs
    threshold = configs['remove_line']['threshold']
    min_line_length = configs['remove_line']['min_line_length']
    max_line_gap = configs['remove_line']['max_line_gap']

    # # Adjust the min_line_length according to the size of image
    # height, width = copy.shape[:2]  # get image size
    # min_line_length = height * 0.9
    # print(min_line_length)

    # find and remove lines
    lines = cv2.HoughLinesP(copy, 1, np.pi / 180, threshold, min_line_length, max_line_gap)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]  # get end point of line : ( (x1, y1) , (x2, y2) )
            if x1 == x2 or y1 == y2:  # only vertical or parallel lines.
                # remove line drawing black line
                cv2.line(copy, (x1, y1), (x2, y2), (0, 0, 0), 10)
    return copy


def get_threshold(image_gray):
    """ 이미지에 Threshold 를 적용해서 흑백(Binary) 이미지객체를 반환합니다.
    이 때 인자로 입력되는 이미지는 Gray-scale 이 적용된 2차원 이미지여야 합니다.
    configs 에 적용된 threshold mode 에 따라 global threshold / mean adaptive threshold / gaussian adaptive threshold
    를 적용할 수 있습니다.

    :param image_gray: Gray-scale 이 적용된 OpenCV image (2 dimension)
    :return: Threshold 를 적용한 흑백(Binary) 이미지
    """
    copy = image_gray.copy()  # copy the image to be processed
    # get configs
    global configs
    mode = configs['threshold']['mode']  # get threshold mode (mean or gaussian or global)
    block_size = configs['threshold']['block_size']
    subtract_val = configs['threshold']['subtract_val']

    if mode == 'mean':  # adaptive threshold - mean
        image_threshold = cv2.adaptiveThreshold(copy, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                                cv2.THRESH_BINARY_INV, block_size, subtract_val)
    elif mode == 'gaussian':  # adaptive threshold - gaussian
        image_threshold = cv2.adaptiveThreshold(copy, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                cv2.THRESH_BINARY_INV, block_size, subtract_val)
    else:  # (mode == 'global') global threshold - otsu's binary operation
        image_threshold = get_otsu_threshold(copy)

    return image_threshold  # Returns the image with the threshold applied.


def get_global_threshold(image_gray, threshold_value=130):
    """ 이미지에 Global Threshold 를 적용해서 흑백(Binary) 이미지객체를 반환합니다.
    하나의 값(threshold_value)을 기준으로 이미지 전체에 적용하여 Threshold 를 적용합니다.
    픽셀의 밝기 값이 기준 값 이상이면 흰색, 기준 값 이하이면 검정색을 적용합니다.
    이 때 인자로 입력되는 이미지는 Gray-scale 이 적용된 2차원 이미지여야 합니다.
    
    :param image_gray:
    :param threshold_value: 이미지 전체에 Threshold 를 적용할 기준 값.
    :return: Global Threshold 를 적용한 흑백(Binary) 이미지
    """
    copy = image_gray.copy()  # copy the image to be processed
    _, binary_image = cv2.threshold(copy, threshold_value, 255, cv2.THRESH_BINARY)
    return binary_image


def get_otsu_threshold(image_gray):
    """  이미지에 Global Threshold 를 적용해서 흑백(Binary) 이미지객체를 반환합니다.
    하나의 값을 기준으로 이미지 전체에 적용하여 Threshold 를 적용합니다.
    해당 값은 Otsu's Binarization 에 의해 자동으로 이미지의 히스토그램을 분석한 후 중간값으로 설정됩니다.
    픽셀의 밝기 값이 기준 값 이상이면 흰색, 기준 값 이하이면 검정색을 적용합니다.
    이 때 인자로 입력되는 이미지는 Gray-scale 이 적용된 2차원 이미지여야 합니다.

    :param image_gray: Gray-scale 이 적용된 OpenCV image (2 dimension)
    :return: Otsu's Binarization에 의해 Global Threshold 를 적용한 흑백(Binary) 이미지
    """
    copy = image_gray.copy()  # copy the image to be processed
    blur = cv2.GaussianBlur(copy, (5, 5), 0)  # Gaussian blur 를 통해 noise 를 제거한 후
    # global threshold with otsu's binarization
    ret3, image_otsu = cv2.threshold(copy, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return image_otsu


def get_closing(image_gray):
    """ 이미지에 Morph Close 를 적용한 이미지객체를 반환합니다.
    이미지에 Dilation 수행을 한 후 Erosion 을 수행한 것입니다.
    이 때 인자로 입력되는 이미지는 Gray-scale 이 적용된 2차원 이미지여야 합니다.
    configs 에 의해 kernel size 값을 설정할 수 있습니다.

    todo +++++++++ 아래 설명은 README 로 옮기자
    커널은 Image Transformation 을 결정하는 구조화된 요소이며
    커널의 크기가 크거나, 반복횟수가 많아지면 과하게 적용되어 경계가 없어질 수도 있습니다.

    :param image_gray: Gray-scale 이 적용된 OpenCV image (2 dimension)
    :return: Morph Close 를 적용한 흑백(Binary) 이미지
    """
    copy = image_gray.copy()  # copy the image to be processed
    # get configs
    global configs
    kernel_size_row = configs['close']['kernel_size_row']
    kernel_size_col = configs['close']['kernel_size_col']
    # make kernel matrix for dilation and erosion
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size_row, kernel_size_col))
    # closing (dilation and erosion)
    image_close = cv2.morphologyEx(copy, cv2.MORPH_CLOSE, kernel)
    return image_close


def get_contours(image):
    """ 이미지에서 Contour 를 추출하여 반환합니다.
    Contour 추출 모드는 configs 에서 설정할 수 있습니다.
    찾은 contour 리스트를 dictionary 형태로 반환합니다.
    이미지 처리(Image processing) 단계를 거친 후 contour 를 잘 추출할 수 있습니다.

    :param image: OpenCV의 image 객체 (2 dimension)
    :return: 이미지에서 추출한 contours
    """
    # get configs
    global configs
    retrieve_mode = configs['contour']['retrieve_mode']  # integer value
    approx_method = configs['contour']['approx_method']  # integer value
    # find contours from the image
    _, contours, _ = cv2.findContours(image, retrieve_mode, approx_method)
    return contours


def draw_contour_rect(image_origin, contours):
    """ 사각형의 Contour 를 이미지 위에 그려서 반환합니다.
    찾은 Contours 를 기반으로 이미지 위에 각 contour 를 감싸는 외각 사각형을 그립니다.

    :param image_origin: OpenCV의 image 객체
    :param contours: 이미지 위에 그릴 contour 리스트
    :return: 사각형의 Contour 를 그린 이미지
    """
    rgb_copy = image_origin.copy()  # copy the image to be processed
    # get configs
    global configs
    min_width = configs['contour']['min_width']
    min_height = configs['contour']['min_height']
    # Draw bounding rectangles
    for contour in contours:
        x, y, width, height = cv2.boundingRect(contour)  # top-left vertex coordinates (x,y) , width, height
        # Draw images that are larger than the standard size
        if width > min_width and height > min_height:
            cv2.rectangle(rgb_copy, (x, y), (x + width, y + height), (0, 255, 0), 2)

    return rgb_copy


def get_cropped_images(image_origin, contours):
    """ 이미지에서 찾은 Contour 부분들을 잘라내어 반환합니다.
    각 contour 를 감싸는 외각 사각형에 여유분(margin)을 주어 이미지를 잘라냅니다.

    :param image_origin: 원본 이미지
    :param contours: 잘라낼 contour 리스트
    :return: contours 를 기반으로 잘라낸 이미지(OpenCV image 객체) 리스트
    """
    image_copy = image_origin.copy()  # copy the image to be processed
    # get configs
    global configs
    min_width = configs['contour']['min_width']
    min_height = configs['contour']['min_height']
    margin = 10  # to give the margin when cropping the images
    origin_height, origin_width = image_copy.shape[:2]  # get image size
    cropped_images = [image_copy]  # list to save the crop image.

    for contour in contours:  # Crop the images with on bounding rectangles of contours
        x, y, width, height = cv2.boundingRect(contour)  # top-left vertex coordinates (x,y) , width, height
        # images that are larger than the standard size
        if width > min_width and height > min_height:
            # The range of row to crop (with margin)
            row_from = (y - margin) if (y - margin) > 0 else y
            row_to = (y + height + margin) if (y + height + margin) < origin_height else y + height
            # The range of column to crop (with margin)
            col_from = (x - margin) if (x - margin) > 0 else x
            col_to = (x + width + margin) if (x + width + margin) < origin_width else x + width
            # Crop the image with Numpy Array
            cropped = image_copy[row_from: row_to, col_from: col_to]
            cropped_images.append(cropped)  # add to the list
    return cropped_images


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


def save_image(image, name_prefix):
    """ 이미지(OpenCV image 객체)를 이미지파일(.jpg)로 저장합니다.

    :param image: 저장할 이미지 (OpenCV image 객체)
    :param name_prefix: 파일명을 식별할 접두어 (확장자 제외)
    :return:
    """
    # make file name with the datetime suffix.
    d_date = datetime.datetime.now()  # get current datetime
    current_datetime = d_date.strftime("%Y%m%d%I%M%S")  # datetime to string
    file_path = name_prefix + "_" + current_datetime + ".jpg"  # complete file name
    cv2.imwrite(file_path, image)


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


def image_to_text_file(image, title, f_stream=None):
    """ OCR 엔진(pytesseract) 를 이용해 이미지에서 글자를 추출하여 파일에 씁니다.

    :param image: 텍스트(Text)를 추출할 resource 이미지
    :param title: 현재 추출한 글자를 구분할 제목 (String)
    :param f_stream: 추출한 텍스트(Text)를 쓸 파일스트림
    :return:
    """
    img = Image.fromarray(image)
    text = ocr.image_to_string(img, lang='kor')
    if f_stream is not None:
        f_stream.write("================ " + title + " ================ \n")
        f_stream.write(text + "\n")
    else:
        print("================ OCR result : " + title + "================")
        print(text)


def get_text_from_image(image):
    """ OCR 엔진(tesseract) 를 이용해 이미지에서 글자를 추출합니다.

    :param image: 텍스트(Text)를 추출할 resource 이미지
    :return: 추출한 텍스트(Text)를 String 형으로 반환
    """
    img = Image.fromarray(image)
    text = ocr.image_to_string(img, lang='kor')
    return text


def process_image(image_file):
    """ 다섯 단계의 이미지 처리(Image precessing)를 힙니다.
    현재 함수에서 순서를 변경하여 적용할 수 있습니다.
    1) Gray-scale 적용
    2) Morph Gradient 적용
    3) Long Line Removal 적용
    4) Threshold 적용
    5) Close 적용

    :param image_file: 이미지 처리(Image precessing)를 적용할 이미지 파일
    :return: 이미지 처리(Image precessing)가 적용된 이미지 (OpenCV image 객체)
    """
    image_origin = open_original(image_file)
    image_origin = cv2.pyrUp(image_origin)  # size up ( x4 )
    # Grey-Scale
    image_gray = get_gray(image_origin)
    # Morph Gradient
    image_gradient = get_gradient(image_gray)
    # Long line remove
    image_line_removed = remove_long_line(image_gradient)
    # Threshold
    image_threshold = get_threshold(image_line_removed)
    # Morph Close
    image_close = get_closing(image_threshold)
    contours = get_contours(image_close)

    return get_cropped_images(image_origin, contours)


def main():
    read_configs('config.yml')
    print_configs()
    cropped_images = process_image('images/test (1).jpg')
    count = 0
    for crop_image in cropped_images:
        count += 1
        save_image(crop_image, "crop_" + str(count))


if __name__ == "__main__":
    main()
