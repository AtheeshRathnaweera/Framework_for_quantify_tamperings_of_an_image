from skimage.metrics import structural_similarity
import cv2 as cv
import glob
from matplotlib import pyplot as plt
import matplotlib.transforms as mtrans


class QuantifyDifference:
    original_image = cv.imread('./original-image/lenna-original.png', 0)
    attacked_files = []

    def compare_on_each_channel(self):
        print("\nOn each channel together started")
        colored_image = cv.imread('./original-image/lenna-original.png')

        # get each channel
        original_blue_channel, original_green_channel, original_red_channel = cv.split(colored_image)

        for fileName in glob.glob("./attacked-images/*"):
            file_name = fileName.split("attacked-images")[1]
            image = cv.imread(fileName)
            image_blue_channel, image_green_channel, image_red_channel = cv.split(image)

            resized_blue_image = cv.resize(image_blue_channel, original_blue_channel.shape)
            resized_green_image = cv.resize(image_green_channel, original_green_channel.shape)
            resized_red_image = cv.resize(image_red_channel, original_red_channel.shape)

            ssim_blue_result = self.ssim_result(resized_blue_image)
            ssim_green_result = self.ssim_result(resized_green_image)
            ssim_red_result = self.ssim_result(resized_red_image)

            print("IMAGE : " + file_name + " SSIM | BLUE : {}".format(ssim_blue_result) + " | GREEN : {}".format(
                ssim_green_result) + " | RED : {}".format(ssim_red_result))

    def compare(self):
        print("\nall channels together started\n")

        for fileName in glob.glob("./attacked-images/*"):
            image = cv.imread(fileName, 0)

            file_name = fileName.split("attacked-images")[1]

            # compute the Structural Similarity Index (SSIM) between the two
            # images, ensuring that the difference image is returned
            ssim_result = self.ssim_result(image)

            print("IMAGE : " + file_name + " | SSIM: {}%".format(round(ssim_result * 100, 2)))

            fig = plt.figure("SSIM Result")

            ax01 = plt.subplot(121), plt.imshow(cv.imread('./original-image/lenna-original.png')[...,::-1])
            plt.title('Original'), plt.xticks([]), plt.yticks([])
            plt.axis("off")

            ax02 = plt.subplot(122), plt.imshow(cv.imread(fileName)[...,::-1])
            plt.title('Tampered'), plt.xticks([]), plt.yticks([])

            trans = mtrans.blended_transform_factory(fig.transFigure,
                                                     mtrans.IdentityTransform())

            txt = fig.text(.5, 60, "SSIM Value : {}%".format(round(ssim_result * 100, 2)), ha='center')
            txt.set_transform(trans)

            plt.show()

    def ssim_result(self, image):
        resized_image = cv.resize(image, self.original_image.shape)

        return structural_similarity(self.original_image, resized_image, multichannel=True,
                                     gaussian_weights=True, sigma=1.5, use_sample_covariance=False,
                                     data_range=255)


main = QuantifyDifference()

main.compare()
# main.compare_on_each_channel()

cv.waitKey(0)
