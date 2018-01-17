

from pre_process import *


def get_progress_image(resource_dir, filename_prefix):
    resource = resource_dir + filename_prefix + ".jpg"  # complete resource image path
    image_origin = open_original(resource)
    image_origin = cv2.pyrUp(image_origin)  # size up ( x4 )
    comparing_images = []

    # Grey-Scale
    image_gray = get_gray(image_origin)
    contours = get_contours(image_gray)
    image_with_contours = draw_contour_rect(image_origin, contours)

    compare_set = merge_vertical(image_gray, image_with_contours)
    comparing_images.append(compare_set)

    # Morph Gradient
    image_gradient = get_gradient(image_gray)
    contours = get_contours(image_gradient)
    image_with_contours = draw_contour_rect(image_origin, contours)

    compare_set = merge_vertical(image_gradient, image_with_contours)
    comparing_images.append(compare_set)

    # Long line remove
    image_line_removed = remove_long_line(image_gradient)
    contours = get_contours(image_line_removed)
    image_with_contours = draw_contour_rect(image_origin, contours)

    compare_set = merge_vertical(image_line_removed, image_with_contours)
    comparing_images.append(compare_set)

    # Threshold
    image_threshold = get_threshold(image_line_removed)
    contours = get_contours(image_threshold)
    image_with_contours = draw_contour_rect(image_origin, contours)

    compare_set = merge_vertical(image_threshold, image_with_contours)
    comparing_images.append(compare_set)

    # Morph Close
    image_close = get_closing(image_threshold)
    contours = get_contours(image_close)
    image_with_contours = draw_contour_rect(image_origin, contours)

    compare_set = merge_vertical(image_close, image_with_contours)
    comparing_images.append(compare_set)

    # Merge all step's images
    image_merged_all = np.hstack(comparing_images)
    # show_window(image_merged_all, 'image_merged_all')  # show all step
    # save_image(image_merged_all, filename_prefix)  # save all step image as a file
    # # save final result
    # save_image(image_with_contours, filename_prefix + '_final_')
    return get_cropped_images(image_origin, contours)


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
            image_to_text_file(crop_image, filename_prefix + "crop_" + str(count), f)
        f.close()


def main():


if __name__ == "__main__":
    main()
