import sys
import cv2
import logging as log
import argparse
import numpy as np
sys.path.append('../src')
from imagefilter import ImageFilter

def build_argparse():
    parser = argparse.ArgumentParser()
    parser.add_argument("-path", type= str,help = "Path to image")
    parser.add_argument("-gray", type=bool, help="Grayscale img" , default=False, required= False)
    parser.add_argument("-shape", type=bool, help="Reshape img",default = False, required= False)
    parser.add_argument("-crop", type=bool, help="Crop img", default = False, required= False)
    parser.add_argument("-width", type=int, help="Image width", default= 300, required= False)
    parser.add_argument("-height", type=int, help="Image height", default =300, required= False)
    parser.add_argument("-X1", type=int, help="X1 coordinate of image", default=100, required=False)
    parser.add_argument("-X2", type=int, help="X2 coordinate of image", default=300, required=False)
    parser.add_argument("-Y1", type=int, help="Y1 coordinate of image", default=100, required=False)
    parser.add_argument("-Y2", type=int, help="Y2 coordinate of image", default=300, required=False)
    parser.add_argument("-alpha", type=int, help="alpha indent", default=100, required=False)
    return parser

def main():
    args = build_argparse().parse_args()
    image = cv2.imread(args.path)
    cv2.imshow("Image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    Filter = ImageFilter(args.gray,args.shape, args.crop)
    result = Filter.process_image(image,args.width,args.height,args.X1,args.X2,args.Y1,args.Y2,args.alpha)
    cv2.imshow("Result",result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return
    
if __name__ == '__main__':
    ImageFilter()
    sys.exit(main())