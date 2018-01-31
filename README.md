# PyTextGotcha


## pre_process.py

#### Introduction

이미지에서 글자 부분을 추출하기 위해 이미지 처리(Image precessing)를 합니다.

* 이미지에서 글자로 추정되는 부분을 찾아 표시하거나 해당 영역을 잘라낼 수 있습니다.

* 글자로 추정되는 부분을 더 잘 찾기 위해 이미지 처리(Image precessing)를 거칩니다.

* 이미지 처리(Image precessing)는 5단계로 구성됩니다.
  process_image() 에서 순서를 변경하여 적용할 수 있습니다.
  * 1. Gray-scale 적용
  * 2. Morph Gradient 적용
  * 3. Long Line Removal 적용
  * 4. Threshold 적용
  * 5. Close 적용


*  위 이미지 처리(Image precessing) 단계를 거친 후 Contour(경계영역)를 추출하면
  글자로 추정되는 영역을 발견할 수 있습니다.

* 각 단계의 함수 내부에서 사용되는 상수 값들은 configs.yml 파일에서 설정가능합니다.

* 아래 import 목록에 해당하는 파이썬 패키지가 개발환경에 설치되어야 합니다.특히 OCR(문자 인식)을 위한 pytesseract 를 사용하기 위해서는 각 언어팩을 함께 설치해야함에 주의하세요. 이미지처리에는 주로 OpenCV 라이브러리를 활용합니다.

todo : 각 단게의 개념을 좀 더 자세히 소개하자.
todo : 각 단계를 거쳤을 때 어떤 결과를 얻을 수 있는지 이미지로 표시하자.

##### 1) Gray-scale 적용

    def get_gray(image_origin):

* image 객체를 인자로 받아서 Gray-scale 을 적용한 2차원 이미지 객체로 반환합니다. 이 때 인자로 입력되는 이미지는 BGR 컬러 이미지여야 합니다.

    __:param image_origin:__ OpenCV 의 BGR image 객체 (3 dimension)

    __:return:__ gray-scale 이 적용된 image 객체 (2 dimension)


##### 2) Morph Gradient 적용

* [참고 - Morphological Transformations 의 cv2.morphologyEx() 함수](http://opencv-python.readthedocs.io/en/latest/doc/12.imageMorphological/imageMorphological.html?highlight=erosion#opening-closing)

---

    def get_gradient(image_gray):
* 이미지에 Dilation 과 Erosion 을 적용하여 그 차이를 이용해 윤곽선을 추출합니다. 이 때 인자로 입력되는 이미지는 Gray scale 이 적용된 2차원 이미지여야 합니다.
* configs.yaml 파일에서 kernel size (kernel_size_row, kernel_size_col)를 설정하여 이미지처리 효과 정도를 조절할 수 있습니다.


    __:param image_gray:__ Gray-scale 이 적용된 OpenCV image (2 dimension)

    __:return:__ 윤곽선을 추출한 결과 이미지 (OpenCV image)



##### 3) Threshold 적용
* [참고 - 이미지 임계처리](http://opencv-python.readthedocs.io/en/latest/doc/09.imageThresholding/imageThresholding.html)

---
    def get_threshold(image_gray):
* 이미지에 Threshold 를 적용해서 흑백(Binary) 이미지객체를 반환합니다.
    이 때 인자로 입력되는 이미지는 Gray-scale 이 적용된 2차원 이미지여야 합니다.
* configs.yaml 파일에서 변수값(mode, block_size, subtract_val)를 설정하여 이미지처리 효과 정도를 조절할 수 있습니다.


    __:param image_gray:__ Gray-scale 이 적용된 OpenCV image (2 dimension)

    __:return:__ Threshold 를 적용한 흑백(Binary) 이미지


* configs 에 적용된 threshold mode 에 따라 아래와 같이 Threshold 를 적용할 수 있습니다.
  * global threshold
  * mean adaptive threshold
  * gaussian adaptive threshold




##### 4) Long Line Removal 적용
* [참고 - Line Detection(허프 변환)](http://opencv-python.readthedocs.io/en/latest/doc/25.imageHoughLineTransform/imageHoughLineTransform.html)

---

    def remove_long_line(image_binary):
* 이미지에서 직선을 찾아서 삭제합니다. 글자 경계를 찾을 때 방해가 되는 직선을 찾아서 삭제합니다. 이 때 인자로 입력되는 이미지 2 차원(2 dimension) 흑백(Binary) 이미지여야 합니다.

* 직선을 삭제할 때는 해당 라인을 검정색으로 그려 덮어 씌웁니다.
* configs.yaml 파일에서 변수값(threshold, min_line_length, max_line_gap)를 설정하여 이미지처리 효과 정도를 조절할 수 있습니다.

    __:param image_binary:__ 흑백(Binary) OpenCV image (2 dimension)


    __:return:__ 라인이 삭제된 이미지 (OpenCV image)





##### 5) Morph Close 적용
* [참고 - Morphological Transformations 의 cv2.morphologyEx() 함수](http://opencv-python.readthedocs.io/en/latest/doc/12.imageMorphological/imageMorphological.html?highlight=erosion#opening-closing)

---

    def get_closing(image_gray):
* 이미지에 Morph Close 를 적용한 이미지객체를 반환합니다. 이미지에 Dilation 수행을 한 후 Erosion 을 수행한 것입니다. 이 때 인자로 입력되는 이미지는 Gray-scale 이 적용된 2차원 이미지여야 합니다.
* configs 에 의해 kernel size 값을 설정할 수 있습니다.
* 커널은 Image Transformation 을 결정하는 구조화된 요소이며
 커널의 크기가 크거나, 반복횟수가 많아지면 과하게 적용되어 경계가 없어질 수도 있습니다.
* configs.yaml 파일에서 변수값(kernel_size_row, kernel_size_col)를 설정하여 이미지처리 효과 정도를 조절할 수 있습니다.

    __:param image_gray:__ Gray-scale 이 적용된 OpenCV image (2 dimension)

    __:return:__ Morph Close 를 적용한 흑백(Binary) 이미지


##### 6) Contour 추출
* [참고 - Image Contours ](http://opencv-python.readthedocs.io/en/latest/doc/15.imageContours/imageContours.html?highlight=contour)

---

    def get_contours(image):
* 이미지에서 Contour 를 추출하여 반환합니다.찾은 contour 리스트를 dictionary 형태로 반환합니다.이미지 처리(Image processing) 단계를 거친 후 contour 를 잘 추출할 수 있습니다.
* configs.yaml 파일에서 변수값(min_width, min_height, retrieve_mode, approx_method)를 설정하여 이미지처리 효과 정도를 조절할 수 있습니다.

    __:param image:__ OpenCV의 image 객체 (2 dimension)

    __:return:__ 이미지에서 추출한 contours









-----------------------------------------------







## preprocess_test.py

#### Introduction
pre_precess.py 에서 정의된 이미지 처리(Image precessing)의
각 단계별 결과 및 최종 결과물에 대하여 테스트하고 분석할 수 있습니다.

* 윈도우를 띄어서 진행단계의 이미지를 확인할 수 있습니다.
* 이미지들을 양옆 및 위아래로 병합해여 비교할 수 있습니다.


##### 이미지 윈도우 열기

    def show_window(image, title='untitled', max_height=700):

* 이미지 윈도우를 열어서 보여줍니다.
  * __:param image:__ 보여줄 이미지 (OpenCV image 객체)
  * __:param title:__ 윈도우 제목
  * __:param max_height:__ 이미지 윈도우 사이즈의 최대 높이




###  이미지 합치기
    def merge_horizontal(image_gray, image_bgr):
* Height 사이즈가 같은 두 이미지를 옆으로(Horizontally) 병합 합니다.이미지 처리(Image processing) 단계를 원본과 비교하기위한 목적으로, 2차원(2 dimension) 흑백 이미지와 3차원(3 dimension) BGR 컬리 이미지를 인자로 받아 병합합니다.

  * __:param image_gray:__ 2차원(2 dimension) 흑백 이미지
  * __:param image_bgr:__ 3차원(3 dimension) BGR 컬리 이미지
  * __:return:__ 옆으로(Horizontally) 병합된 이미지

---

    def merge_vertical(image_gray, image_bgr):
*  Width 사이즈가 같은 두 이미지를 위아래로(Vertically) 병합 합니다. 이미지 처리(Image processing) 단계를 원본과 비교하기위한 목적으로, 2차원(2 dimension) 흑백 이미지와 3차원(3 dimension) BGR 컬리 이미지를 인자로 받아 병합합니다.
  * __:param image_gray:__ 2차원(2 dimension) 흑백 이미지
  * __:param image_bgr:__ 3차원(3 dimension) BGR 컬리 이미지
  * __:return:__ 위아래로(Vertically) 병합된 이미지

---


### 전체 단계별 이미지처리 결과 한 눈에 확인하기

    def get_step_compare_image(path_of_image):

* 이미지 프로세싱 전 단계의 중간 결과물을 하나로 병합하여 반환합니다.

  * __:param path_of_image:__ 이미지 처리(Image precessing)를 적용할 이미지 파일
  * __ :return:__ 이미지 처리 단계별 결과를 한 눈에 확인할 수 있도록 모두 병합한 이미지를 반환











-----------------------------------------------










## config.yml

    threshold:
        block_size: 9 # Threshold (Odd number !!)
        subtract_val: 12  # Threshold

* block_size 는 Odd number(홀수)여야 합니다.


---

    gradient:
        kernel_size_row: 2  # Gradient Kernel Size
        kernel_size_col: 2  # Gradient Kernel Size

*

---

    close:
        kernel_size_row: 2  # Closing Kernel Size
        kernel_size_col: 2  # Closing Kernel Size

---

    remove_line:
        threshold: 100  # Long Line Remove Precision
        min_line_length: 100  # Long Line Remove  Minimum     Line Length
        max_line_gap: 5  # Long Line Remove Maximum Line Gap

---

    contour:
        min_width: 4  # Minimum Contour Rectangle Size
        min_height: 10  # Minimum Contour Rectangle Size
        retrieve_mode: 0  # RETR_EXTERNAL = 0. RETR_LIST = 1, RETR_CCOMP = 2, RETR_TREE = 3, RETR_FLOODFILL = 4
        approx_method: 2  # CHAIN_APPROX_NONE = 1, CHAIN_APPROX_SIMPLE = 2, CHAIN_APPROX_TC89_KCOS = 4, CHAIN_APPROX_TC89_L1 = 3

*  retrieve_mode
  * cv2.RETR_EXTERNAL : contours line중 가장 바같쪽 Line만 찾음.
  * cv2.RETR_LIST : 모든 contours line을 찾지만, hierachy 관계를 구성하지 않음.
  * cv2.RETR_CCOMP : 모든 contours line을 찾으며, hieracy관계는 2-level로 구성함.
  * cv2.RETR_TREE : 모든 contours line을 찾으며, 모든 hieracy관계를 구성함.

* approx_method
  * cv2.CHAIN_APPROX_NONE : 모든 contours point를 저장.
  * cv2.CHAIN_APPROX_SIMPLE : contours line을 그릴 수 있는 point 만 저장. (ex; 사각형이면 4개 point)
  * cv2.CHAIN_APPROX_TC89_L1 : contours point를 찾는 algorithm
  * cv2.CHAIN_APPROX_TC89_KCOS : contours point를 찾는 algorithm

---






-----------------------------------------------





## judge_text.py

#### Introduction

Inception v3 모델을 가지고 Transfer Learning 을 통해 새로운 모델을 만들었습니다. 모델을 활용해서 input 이미지가 Text인지 Text가 아닌지 두 종류의 이미지분류 결과를 알려줍니다.

* [참고- 텐서플로우 - 이미지 분류 (Inception 모델 이용하기)](https://blog.naver.com/PostView.nhn?blogId=flowerdances&logNo=221192170996&parentCategoryNo=&categoryNo=32&viewDate=&isShowPopularPosts=false&from=postView)

#### Useage

     >python judge_text.py some_directory_path/test_image.jpg

*  결과는 아래와 같이 출력됩니다.

---

    text (score = 0.96235)
    not text (score = 0.03765)










-----------------------------------------------





## References


### Naver D2 blog post

* [딥러닝과 OpenCV를 활용해 사진 속 글자 검출하기](http://d2.naver.com/helloworld/8344782)

  * 본문에서 사용한 configure value 는 아래와 같다고 한다.
    * gradient시 Kernel size는2 x 2.
    * close시 Kernel size는 9 x 5.
    * threshold시 block size는 3을 사용했습니다.


### StackOverflow
* [Extracting text OpenCV ?](https://stackoverflow.com/questions/23506105/extracting-text-opencv/23672571#23672571)
  * openCV-Python 을 이용해 비슷한 방식으로 명함의 text를 뽑아냄



