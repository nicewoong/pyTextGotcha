# PyTextGotcha




## Components

### pre_process.py
* 1) Gray Scale 적용
* 2) Gradient 추출
* 3) Adaptive Threshold 적용
* 4) Close 적용
* 5) Long Line remove 적용
* 6) 위 단계를 모두 거친 image 로부터 Contour 추출
* 7) Contours 들 중 최소 사이즈 이상인 것들만 사각형(Rectangle)으로 원본이미지에 그리기
* 8) Rectangle 표시된 부분 자르기


### text decision (Will be implemented soon)

* Determine whether it is text or not using CNN (Deep learnig)


### config.yml

    threshold:
        block_size: 9 # Threshold (Odd number !!)
        subtract_val: 12  # Threshold

    gradient:
        kernel_size_row: 2  # Gradient Kernel Size
        kernel_size_col: 2  # Gradient Kernel Size

    close:
        kernel_size_row: 2  # Closing Kernel Size
        kernel_size_col: 2  # Closing Kernel Size

    remove_line:
        threshold: 100  # Long Line Remove Precision
        min_line_length: 100  # Long Line Remove  Minimum     Line Length
        max_line_gap: 5  # Long Line Remove Maximum Line Gap

    contour:
        min_width: 4  # Minimum Contour Rectangle Size
        min_height: 10  # Minimum Contour Rectangle Size
        retrieve_mode: 0  # RETR_EXTERNAL
                        # RETR_EXTERNAL = 0
                        # RETR_LIST = 1
                        # RETR_CCOMP = 2
                        # RETR_TREE = 3
                        # RETR_FLOODFILL = 4
        approx_method: 2  # CHAIN_APPROX_SIMPLE
                        # CHAIN_APPROX_NONE = 1
                        # CHAIN_APPROX_SIMPLE = 2
                        # CHAIN_APPROX_TC89_KCOS = 4
                        # CHAIN_APPROX_TC89_L1 = 3



## References


### Naver D2 blog post

* [딥러닝과 OpenCV를 활용해 사진 속 글자 검출하기](http://d2.naver.com/helloworld/8344782)
* 본문에서 사용한 configure value 는 아래와 같다고 한다.

> gradient시 Kernel size는2 x 2.

> close시 Kernel size는 9 x 5.

> threshold시 block size는 3을 사용했습니다.


### StackOverflow
* [Extracting text OpenCV ?](https://stackoverflow.com/questions/23506105/extracting-text-opencv/23672571#23672571)
* openCV-Python 을 이용해 비슷한 방식으로 명함의 text를 뽑아냄



