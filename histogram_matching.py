import glob
import cv2 as cv
from skimage.metrics import structural_similarity
from matplotlib import pyplot as plt
import matplotlib.transforms as mtrans


class HistogramMatching:
    original_image = cv.imread('./original-image/lenna-original.png')

    def __init__(self):
        print("started")

    def extract_diff(self, base_image_gray, check_image_gray):
        resized_image = cv.resize(check_image_gray, base_image_gray.shape)

        (score, diff) = structural_similarity(base_image_gray, resized_image, full=True)

        # The diff image contains the actual image differences between the two images
        # and is represented as a floating point data type in the range [0,1]
        # so we must convert the array to 8-bit unsigned integers in the range
        # [0,255] image1 we can use it with OpenCV
        return (diff * 255).astype("uint8")

    def compare(self):
        # hue varies from 0 to 179, saturation from 0 to 255
        h_ranges = [0, 180]
        s_ranges = [0, 256]
        ranges = h_ranges + s_ranges

        for fileName in glob.glob("./attacked-images/*"):
            file_name = fileName.split("attacked-images")[1]
            image = cv.imread(fileName)

            base_hsv = cv.cvtColor(self.original_image, cv.COLOR_BGR2HSV)
            check_hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)

            # Calculate the Hist for each images
            hist_base = cv.calcHist(base_hsv, [0, 1], None, [180, 256], ranges)
            cv.normalize(hist_base, hist_base, 0, 255, cv.NORM_MINMAX)
            hist_check = cv.calcHist(check_hsv, [0, 1], None, [180, 256], ranges)
            cv.normalize(hist_check, hist_check, 0, 255, cv.NORM_MINMAX)

            # Compare two Hist. and find out the correlation value
            result = cv.compareHist(hist_base, hist_check, 0)

            print("IMAGE : " + file_name + " | VALUE : {}%".format(round(result * 100, 2)))

            fig = plt.figure("FEATURE MAPPING Result")

            ax01 = plt.subplot(121), plt.imshow(cv.imread('./original-image/lenna-original.png')[..., ::-1])
            plt.title('Original'), plt.xticks([]), plt.yticks([])
            plt.axis("off")

            ax02 = plt.subplot(122), plt.imshow(cv.imread('./attacked-images/' +file_name)[..., ::-1])
            plt.title('Tampered'), plt.xticks([]), plt.yticks([])

            trans = mtrans.blended_transform_factory(fig.transFigure,
                                                     mtrans.IdentityTransform())

            txt = fig.text(.5, 60, "Amount of features on the original : {}%".format(round(result * 100, 2)), ha='center')
            txt.set_transform(trans)

            plt.show()

            # if fileName == "./attacked-images\lenna-center-text.jpg":
            #     # Convert images to grayscale
            #
            #     base_image_gray = cv.cvtColor(self.original_image, cv.COLOR_BGR2GRAY)
            #     check_image__gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
            #
            #     resized_image = cv.resize(check_image__gray, base_image_gray.shape)
            #
            #     diff = self.extract_diff(base_image_gray, check_image__gray)
            #
            #     # Threshold the difference image, followed by finding contours to
            #     # obtain the regions of the two input images that differ
            #     thresh = cv.threshold(diff, 0, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)[1]
            #     contours = cv.findContours(thresh.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
            #     contours = contours[0] if len(contours) == 2 else contours[1]
            #
            #     mask = np.zeros(self.original_image.shape, np.uint8)
            #     filled_after = resized_image.copy()
            #
            #     for c in contours:
            #         area = cv.contourArea(c)
            #         if area > 40:
            #             x, y, w, h = cv.boundingRect(c)
            #             cv.rectangle(base_image_gray, (x, y), (x + w, y + h), (36, 255, 12), 2)
            #             cv.rectangle(resized_image, (x, y), (x + w, y + h), (36, 255, 12), 2)
            #             cv.drawContours(mask, [c], 0, (0, 255, 0), -1)
            #             cv.drawContours(filled_after, [c], 0, (0, 255, 0), -1)
            #
            #     cv.imshow('before', base_image_gray)
            #     cv.imshow('after', resized_image)
            #     cv.imshow('diff', diff)
            #     cv.imshow('mask', mask)
            #     cv.imshow('filled after', filled_after)
            #
            #     inpianted = cv.inpaint(resized_image,cv.cvtColor(mask, cv.COLOR_BGR2GRAY), 3, cv.INPAINT_TELEA)
            #     cv.imshow('inpainted', inpianted)
            #
            #     cv.waitKey(0)


HistogramMatching().compare()
# cv.waitKey(0)
