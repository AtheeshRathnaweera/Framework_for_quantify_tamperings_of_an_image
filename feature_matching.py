import glob
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.transforms as mtrans


class FeatureMatching:
    original_image = cv.imread('./original-image/lenna-original.png', 0)
    attacked_files = []

    def __init__(self):
        self.get_the_files()

    def get_the_files(self):
        for fileName in glob.glob("./attacked-images/*"):
            image = cv.imread(fileName, 0)
            self.attacked_files.append({
                "name": fileName.split("attacked-images")[1],
                "image": image
            })

    def matching_and_presenting_result(self, matcher, kps_of_original_image, des_of_original_image, results):
        # matching the images

        for item in results:
            if item["descriptors"] is not None:
                matches = matcher.knnMatch(des_of_original_image, item["descriptors"], k=2)
                good_matches = []
                for m, n in matches:
                    if m.distance < 0.80 * n.distance:
                        good_matches.append([m])

                matching_result_image = cv.drawMatchesKnn(self.original_image, kps_of_original_image, item["image"],
                                                          item["key_points"], good_matches, None, flags=2)
                # cv.imshow(item["name"], matching_result_image)

                match_result = round(len(good_matches) * 100 / len(item["descriptors"]), 2)

                match_result = '----' if match_result > 100.00 else match_result

                print("IMAGE : " + item["name"] + " | GOOD_MATCHES_AMOUNT : " + str(len(good_matches))
                      + " | PERCENTAGE :  {}%".format(match_result))



                fig = plt.figure("FEATURE MAPPING Result")

                ax01 = plt.subplot(121), plt.imshow(cv.imread('./original-image/lenna-original.png')[..., ::-1])
                plt.title('Original'), plt.xticks([]), plt.yticks([])
                plt.axis("off")

                ax02 = plt.subplot(122), plt.imshow(cv.imread('./attacked-images/'+item["name"])[..., ::-1])
                plt.title('Tampered'), plt.xticks([]), plt.yticks([])

                trans = mtrans.blended_transform_factory(fig.transFigure,
                                                         mtrans.IdentityTransform())

                txt = fig.text(.5, 60, "Amount of features on the original : {}%".format(match_result), ha='center')
                txt.set_transform(trans)

                plt.show()

        # draw only key points location,not size and orientation
        # img01View = cv.drawKeypoints(img01, kp, None, color=(0, 255, 0), flags=0)

    def get_the_kps_and_des(self, detector, image):
        kp, des = detector.detectAndCompute(image, None)

        sharpner_kernels = [np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]]),
                            np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])]

        if kp is None or des is None:
            print("kp and des not found")
            sharpen = cv.filter2D(image, -1, sharpner_kernels[1])
            cv.imshow('sharpen', sharpen)
            cv.waitKey()
            kp, des = detector.detectAndCompute(sharpen, None)

        return kp, des

    def use_orb(self):
        # Initiate ORB detector
        # default nFeatures is 500
        orb_for_original = cv.ORB_create(nfeatures=5000, scoreType=cv.ORB_FAST_SCORE)
        orb_for_check = cv.ORB_create(nfeatures=5000, scoreType=cv.ORB_FAST_SCORE)

        kp_or_original, des_of_original = self.get_the_kps_and_des(orb_for_original, self.original_image)

        # img01View = cv.drawKeypoints(self.original_image, kp_or_original, None, color=(0, 255, 0), flags=0)
        # cv.imshow('image-original', img01View)

        orb_result = []

        for file_data in self.attacked_files:
            orb = None
            # resized_image = cv.resize(file_data["image"], self.original_image.shape)

            if file_data["image"].shape == self.original_image.shape:
                orb = orb_for_original
            else:
                orb = orb_for_check

            kp, des = self.get_the_kps_and_des(orb, file_data["image"])

            new_file_data = {
                "name": file_data["name"],
                "image": file_data["image"],
                "key_points": kp,
                "descriptors": des,
                "good_matches": []
            }
            orb_result.append(new_file_data)

        # brute force matchers
        matcher_hamming = cv.BFMatcher(cv.NORM_HAMMING)
        self.matching_and_presenting_result(matcher_hamming, kp_or_original, des_of_original, orb_result)


FeatureMatching().use_orb()
cv.waitKey(0)
