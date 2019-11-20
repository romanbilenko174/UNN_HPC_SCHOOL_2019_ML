"""

Inference engine detector
 
"""
import cv2
import numpy as np
import openvino
from openvino.inference_engine import IENetwork, IECore
from time import time
class InferenceEngineDetector:
    def __init__(self, weightsPath = None, configPath = None,
                 device = 'CPU', extension = None):
        configPath = 'C:/UNN_HPC_SCHOOL_2019_ML/model\public/mobilenet-ssd\FP16/mobilenet-ssd.xml'
        cpu_extension = "C:/Program Files (x86)/IntelSWTools/openvino/inference_engine/bin/intel64/Release/cpu_extension_avx2.dll"
        weightsPath = "C:/UNN_HPC_SCHOOL_2019_ML/model/public/mobilenet-ssd/FP16/mobilenet-ssd.bin"
        self.ie = IECore()
        if cpu_extension:
            self.ie.add_extension(cpu_extension, 'CPU')
        self.net = IENetwork(model=configPath, weights=weightsPath)
        self.exec_net = self.ie.load_network(network=self.net,
                                        device_name='CPU')
        return
    def draw_detection(self, img,p1x,p1y,p2x,p2y, line_width,name):
        img = cv2.rectangle(img, (p1x, p1y), (p2x, p2y), (0, 255, 0), line_width)
        img = cv2.putText(img, name, (p1x, p1y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
        return img
    def prepare_image(self, image, h, w):
        image = cv2.resize(image, (w, h))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.transpose((2, 0, 1))
        image = np.expand_dims(image, axis=0)
        return image
    def detect(self, image):
        input_blob = next(iter(self.net.inputs))
        out_blob = next(iter(self.net.outputs))
        n, c, h, w = self.net.inputs[input_blob].shape
        blob = self.prepare_image(image, h, w)
        output = self.exec_net.infer(inputs={input_blob: blob})
        output = output[out_blob]
        return output
    def quality(self,image,number_iter):
        times = []
        t0_total = time()
        for i in range(number_iter):
            t0 = time()
            output = self.exec_net.infer(...)
            t1 = time()
            times.append(t1 - t0)
        t1_total = time()
        latency = np.median(times)
        fps = np.median (number_iter/ (t1_total - t0_total))