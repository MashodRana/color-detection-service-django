import color_detection


import cv2


class Service:
    def __init__(self, image_path):
        self.image_path = image_path

        # Read image
        self.image = cv2.imread(image_path)

        # Resize image
        self.image = cv2.resize(self.image, (100, 100), interpolation=cv2.INTER_AREA)

        # The image inside a particular area
        self.box_img = None

        # Resultant image after identify the colors.
        self.resultant_image = None

        # define boundary box of the image
        self.bounding_box = (20, 20, self.image.shape[0] - 20, self.image.shape[1] - 20)

    def __call__(self, detector):
        # Extracting contours.
        contours = detector.extract_contours(self.bounding_box)

        # Extracting pixel values of of each contours.
        px_vals = detector.extract_contours_pxs(contours, False)

        # Extracting name of the colors present in the contours
        colors = detector.extract_colors_kmean(num_clusters=5, px_values=px_vals, show_chart=False)

        return colors


# For testing purpose
def extract_color(path):
    im = "/mnt/sda3/web_development_projects/color_detection/images/test25.jpg"
    service = Service(im)
    result = service(color_detection.ColorDetector(service.image))
    return result
    print(result)
