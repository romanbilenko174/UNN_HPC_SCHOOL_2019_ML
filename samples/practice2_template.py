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
    parser.add_argument("-imgpath", type=str, help="Path to image", required=False)
    parser.add_argument("-lw", type=str, help="Width of line", required=False, default = 1)
    parser.add_argument("-vidpath", type=str, help="Path to video", required=False)
    return parser

def main():
    args = build_argparse().parse_args()
    #image = cv2.imread(args.path)
    #height = image.shape[0]
    #width = image.shape[1]
    cap = cv2.VideoCapture(args.vidpath)
    if (cap.isOpened() == False):
        print("Error opening video stream or file")

    # Read until video is completed
    while (cap.isOpened()):
        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret == True:
            # Display the resulting frame
            IE = InferenceEngineDetector()
            result = IE.detect(frame)
            for i in result:
                for j in i:
                    result = j[~(j == 0).all(1)]
            number_iter = len(result)
            classes = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair",
                       "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
            width = cap.get(3)
            height = cap.get(4)
            for i in result:
                print(i)
                break
                name = int(result[0][1])
                name = classes[name]
                xmin = int(result[0][3] * width)
                ymin = int(result[0][4] * height)
                xmax = int(result[0][5] * width)
                ymax = int(result[0][6] * height)
                img = IE.draw_detection(frame, xmin, ymin, xmax, ymax, args.lw, name)
            #cv2.imshow("draw", img)
            # Press Q on keyboard to  exit
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

        # Break the loop
        else:
            break

    # When everything done, release the video capture object
    cap.release()

    # Closes all the frames
    cv2.destroyAllWindows()

    #self.number_iter = 25

    return 

if __name__ == '__main__':
    sys.exit(main()) 