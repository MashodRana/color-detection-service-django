import os
import re
import json
import numpy as np
from collections import Counter

from matplotlib import pyplot as plt
from sklearn.cluster import DBSCAN, KMeans
from sklearn.neighbors import NearestNeighbors
import cv2

import string
import random
from django.conf import settings


class ColorDetector:

    def __init__(self, image):
        self.color_code_file = "color_name_hex_rgb_codes.json"
        self.color_codes = self.load_file()
        self.image = image

        # Convert the image to RGB
        self.rgb_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)

    def load_file(self):
        """ This method used to load the json file. """
        abs_path = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(abs_path, self.color_code_file)
        if os.path.isfile(file_path):
            with open(file_path, 'r') as j:
                color_codes = json.load(j)
            return color_codes
        else:
            raise FileNotFoundError(f"No '{self.color_code_file}' on the following directory '{abs_path}'")

    @staticmethod
    def rgb2hex(rgb):
        """ This method is used to convert RGB value to HEX """
        return "#{:02x}{:02x}{:02x}".format(int(rgb[0]), int(rgb[1]), int(rgb[2]))

    def get_labels(self, rgbs):
        """
            This method is used to label the RGB color codes.
        :param rgbs: contains the rgb values which will be labeled.
        :param color_codes: A dictionary contains the RGB color codes and corresponding color name.
        :return: A list contains the name of the colors
        """
        Y = np.array(self.color_codes['names'])

        neigh = NearestNeighbors(n_neighbors=1, radius=10)  # Nearest Neighbour with n=1
        neigh.fit(self.color_codes['rgbs'])

        X = rgbs
        index = neigh.kneighbors(X, 1, return_distance=False)  # get the most nearest neighbours label

        return Y[index]  # get and return the color label

    def extract_contours(self, bounding_box):
        """
            This method is used to extract contours from an image within a bounding box(rectangle).
        :param bounding_box: A tuple (x1,y1,x2,y2)=> left bottom corner of a rectangle x1,y1 and right
            top corner x2,y2 of the rectangle. Contours will be extracted within this boundary.
        :return: A list which contains the extracted contours.
        """

        blur = cv2.GaussianBlur(self.image, (5, 5), 0)
        gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)

        # convert gray to binary image
        _, bin = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)
        # edged = cv2.Canny(gray, 100, 200)

        # Find contours
        contours, _ = cv2.findContours(bin, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        avg = sum([cv2.contourArea(c) for c in contours]) / len(contours)
        cnts = [c for c in contours if cv2.contourArea(c) >= avg]
        return cnts

    def extract_contours_pxs(self, contours, disp=False):
        """
            This method is used to pixel values form the contours.
        :param contours: A list of contours extracted with OpenCV.
        :param disp: Extracted contours will be display for True.
        :return: A numpy array contains pixel values.
        """

        # Create numpy array with image height and width
        mask = np.zeros(self.image.shape[:2], np.uint8)

        # Draw the contour on the mask
        cv2.drawContours(mask, contours, -1, color=255, thickness=cv2.FILLED)
        mask2 = np.where(mask == 255, 1, 0)
        mask3 = self.rgb_image * mask2[:, :, np.newaxis]
        resultant_image = mask3.copy()

        if disp:
            show_image(mask3)

        # Find the coordinates of pixels and pixels
        coordinates = np.where(mask3 != 0)  # return the (x,y) position of each pixel which has value 1
        px_values = self.rgb_image[coordinates[0], coordinates[1]]  # Extract coordinates value from the image

        return px_values

    def extract_colors_kmean(self, num_clusters, px_values, show_chart=False):
        """
            This method is used to extract colors by using cluster algorithm K-means and labeled them.
        :param num_clusters: A integer tell how many cluster is needed.
        :param px_values: A numpy array contain the pixel values which will be clustered into num_clusters.
        :param show_chart: Displaying all the cluster with pie chart if True.

        :return : A list contains the name of the colors.
        """

        # Apply K-means clustering
        print('Number of cluster is :', num_clusters)
        clf = KMeans(n_clusters=num_clusters)
        labels = clf.fit_predict(px_values)
        center_colors = clf.cluster_centers_  # get the center of the clusters
        # print(f'center colors: {center_colors}')

        counts = Counter(labels)

        # We get ordered colors by iterating through the keys
        c = dict(counts)
        c = sorted(c.items(), key=lambda x: x[1], reverse=True)
        c = {k: v for k, v in c}
        counts = c
        total_px_vals = 0
        for k, v in counts.items():
            total_px_vals += v
        percentage = [v / total_px_vals * 100 for k, v in counts.items()]
        print('Percentage:', percentage)

        ordered_colors = [center_colors[i] for i in counts.keys()]
        hex_colors = [self.rgb2hex(i) for i in ordered_colors]
        rgb_colors = [i for i in ordered_colors]

        # Labeling the cluster
        color_names = self.get_labels(rgb_colors).flatten()

        fig, (ax1, ax2) = plt.subplots(1,2, figsize=(10,6))
        ax1.imshow(self.rgb_image)
        ax1.set(xticks=[], yticks=[])
        ax1.axis('off')
        
        ax2.pie(counts.values(), labels=color_names, colors=hex_colors)
        ax2.set(xticks=[], yticks=[])
        ax2.axis('off')

        #     plt.savefig('./tested_images/test_color_'+str(ii)+'b.png')
        while(True):
            fname = ''.join(random.choices(string.ascii_uppercase + string.digits, k=10)) + '.png'
            fname = os.path.join(settings.MEDIA_ROOT, 'responses', fname)
            if not os.path.exists(fname):
                plt.savefig(fname)
                return color_names, percentage, fname
