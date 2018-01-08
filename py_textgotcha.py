#!/usr/bin/python
# -*- coding: utf-8 -*-
"""  Detect text in the image.
And processes the image to extract the text portions using OpenCV-Python and CNN.

1) 입력 이미지 Gray Scale 적용
2) Gradient 추출 (추후)
3) Adaptive Threshold 적용한 image 반환, params 로 조정 값 받을 수 있도록
4) Close 적용한 image 반환
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




def main():

    return None


if __name__ == "__main__":
    main()
