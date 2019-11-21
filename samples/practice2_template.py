"""

Inference engine detector sample
 
"""
import sys
import cv2
import argparse
import numpy as np
import logging as log
sys.path.append('../src')
from ie_detector import InferenceEngineDetector
def build_argparse():
    parser = argparse.ArgumentParser()
    parser.add_argument("-img", type=bool, help="Work with image", required=False)
    parser.add_argument("-vid", type=str, help="Work with video", required=False)
    parser.add_argument("-imgpath", type=str, help="Path to image", required=False)
    parser.add_argument("-lw", type=str, help="Width of line", required=False, default = 1)
    parser.add_argument("-vidpath", type=str, help="Path to video", required=False)
    args, leftovers = parser.parse_known_args()
    return parser
def cycle(IE,image,width,height,classes,lw):

        result,fps = IE.detect(image)
        for i in result:
            for j in i:
                result = j[~(j == 0).all(1)]
        for i in range(len(result)):
            data = result[i, :]
            if (data[2] > 0.5):
                name = int(data[1])
                name = classes[name]
                xmin = int(data[3] * width)
                ymin = int(data[4] * height)
                xmax = int(data[5] * width)
                ymax = int(data[6] * height)
                image = IE.draw_detection(image, xmin, ymin, xmax, ymax,lw, name,fps)
        return image
def main():
    args = build_argparse().parse_args()
    IE = InferenceEngineDetector()
    classes = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair",
               "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train",
               "tvmonitor"]
    if args.img:
        image = cv2.imread(args.imgpath)
        height = image.shape[0]
        width = image.shape[1]
        image = cycle(IE,image,width,height,classes,args.lw)
        cv2.imshow("Image", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    elif args.vid:
        cap = cv2.VideoCapture(args.vidpath)
        while True:
            ret, frame = cap.read()
            if ret == True:
                width = cap.get(3)
                height = cap.get(4)
                frame= cycle(IE,frame, width, height, classes,args.lw)
                cv2.imshow("Video", frame)
                if cv2.waitKey(25) & 0xFF == ord('q'):
                        break
            else:
                break
        cap.release()
        cv2.destroyAllWindows()
    return
if __name__ == '__main__':
    sys.exit(main()) 