import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from skimage.metrics import structural_similarity


class MarkingAreas:
    # All the 6 methods for comparison in a list
    methods = ['cv.TM_CCOEFF', 'cv.TM_CCOEFF_NORMED', 'cv.TM_CCORR',
               'cv.TM_CCORR_NORMED', 'cv.TM_SQDIFF', 'cv.TM_SQDIFF_NORMED']

    def mark_the_text(self, original_file_name, check_file_name):
        original_image = cv.imread(original_file_name)
        check_image = cv.imread(check_file_name)

        base_image_gray = cv.cvtColor(original_image, cv.COLOR_BGR2GRAY)
        check_image__gray = cv.cvtColor(check_image, cv.COLOR_BGR2GRAY)

        resized_image = cv.resize(check_image__gray, base_image_gray.shape)
        (score, diff) = structural_similarity(base_image_gray, resized_image, full=True)

        # The diff image contains the actual image differences between the two images
        # and is represented as a floating point data type in the range [0,1]
        # so we must convert the array to 8-bit unsigned integers in the range
        # [0,255] image1 we can use it with OpenCV
        diff_converted = (diff * 255).astype("uint8")

        # Threshold the difference image, followed by finding contours to
        # obtain the regions of the two input images that differ
        thresh = cv.threshold(diff_converted, 0, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)[1]
        contours = cv.findContours(thresh.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        contours = contours[0] if len(contours) == 2 else contours[1]

        mask = np.zeros(original_image.shape, np.uint8)
        filled_after = resized_image.copy()

        for c in contours:
            area = cv.contourArea(c)
            if area > 40:
                x, y, w, h = cv.boundingRect(c)
                cv.rectangle(original_image, (x, y), (x + w, y + h), (36, 255, 12), 2)
                cv.rectangle(check_image, (x, y), (x + w, y + h), (36, 255, 12), 2)
                cv.drawContours(mask, [c], 0, (0, 255, 0), -1)
                cv.drawContours(filled_after, [c], 0, (0, 255, 0), -1)

        cv.imshow('Text appeared places on original', original_image)
        # cv.imshow('after', check_image)
        # cv.imshow('diff', diff_converted)
        # cv.imshow('mask', mask)
        # cv.imshow('filled after', filled_after)

        cv.waitKey(0)

    def mark_the_crop_area(self, original_file_name, check_file_name):

        original_image = cv.imread(original_file_name, 0)
        cropped_image = cv.imread(check_file_name, 0)

        w, h = cropped_image.shape[::-1]
        img = original_image.copy()
        meth = eval(self.methods[0])
        # Apply template Matching
        res = cv.matchTemplate(img, cropped_image, meth)
        min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
        # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
        if self.methods[0] in [cv.TM_SQDIFF, cv.TM_SQDIFF_NORMED]:
            top_left = min_loc
        else:
            top_left = max_loc
        bottom_right = (top_left[0] + w, top_left[1] + h)
        cv.rectangle(img, top_left, bottom_right, 255, 2)

        plt.figure(check_file_name)
        plt.subplot(121), plt.imshow(res, cmap='gray')
        plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
        plt.subplot(122), plt.imshow(img, cmap='gray')
        plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
        plt.show()

    def anything(self,original_file_name, check_file_name):
        original_image = cv.imread(original_file_name, 0)
        cropped_image = cv.imread(check_file_name, 0)

        image = original_image.copy()
        gray = cv.cvtColor(cropped_image, cv.COLOR_BGR2GRAY)
        blur = cv.GaussianBlur(gray, (5, 5), 0)
        thresh = cv.threshold(blur, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)[1]

        cnts = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        for c in cnts:
            area = cv.contourArea(c)
            if area > 10000:
                cv.drawContours(image, [c], -1, (36, 255, 12), 3)

        cv.imwrite('thresh.png', thresh)
        cv.imwrite('image.png', image)
        cv.waitKey(0)


main = MarkingAreas()
# main.mark_the_crop_area('./original-image/lenna-original.png', './attacked-images/lenna-cropped.png')
# main.mark_the_crop_area('./original-image/lenna-original.png', './attacked-images/lenna-cropped-rotate.png')

main.mark_the_text('./original-image/lenna-original.png', './attacked-images/lenna-center-text.jpg')
