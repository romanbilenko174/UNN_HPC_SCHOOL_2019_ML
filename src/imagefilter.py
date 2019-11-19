import cv2
import argparse
class ImageFilter():
    def __init__(self, gray = False, shape = None, crop = None):
        self.gray = gray
        self.shape = shape
        self.crop = crop

    def build_argparse():
        parser = argparse.ArgumentParser()
        parser.add_argument("-path", type=str, help="Path to image")
        return parser

    def process_image(self, image, width,height,X1,X2,Y1,Y2,alpha):
        if self.gray:
            image= cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.shape:
            dim = (width, height)
            image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
        if self.crop:
            image = image[X1:X2+alpha , Y1:Y2 + alpha]
        return image