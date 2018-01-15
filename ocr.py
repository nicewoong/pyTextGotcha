# -*- coding: utf-8 -*-
""" This is a practice code to learn the concept of OCR
"""
__author__ = "Wooongje Han"

from PIL import Image
from pytesseract import *


def orc_test():
    filename = 'images/test3.jpg'
    image = Image.open(filename)
    text = image_to_string(image, lang='kor')

    print("================ OCR result ================")
    print(text)


def main():
    orc_test()
    return None


if __name__ == "__main__":
    main()