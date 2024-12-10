from abc import ABC, abstractmethod

import cv2
import numpy as np

from ultralytics import YOLO
import onnxruntime as ort

class ModelWrapper(ABC):
    def __init__(self, name_: str):
        self.name = f'models/{name_}'
        self.names = YOLO(f'{self.name}.onnx').names
        self.input  = []
        self.output = []
    @abstractmethod
    def inference(self):
        pass
    @abstractmethod
    def drawResult(self):
        pass
    @staticmethod
    def pt2onnx():
        model = YOLO("models/yolo11n.pt")
        model.export(format="onnx", imgsz=[640,640], opset=12)

class onnxModel(ModelWrapper):
    def __init__(self, name_: str):
        super().__init__(name_)
    @abstractmethod
    def inference(self):
        pass
    def drawResult(self):
        out = self.output
        class_ids, confs, boxes = list(), list(), list()
        rows = out[0].shape[0] 
        for i in range(rows):
            row = out[0][i]
            conf = row[4]
            classes_score = row[4:]
            _,_,_, max_idx = cv2.minMaxLoc(classes_score)
            class_id = max_idx[1]
            if (classes_score[class_id] > 0): # > .15):
                confs.append(conf)
                label = int(class_id)
                class_ids.append(label)
                x, y, w, h = row[0].item(), row[1].item(), row[2].item(), row[3].item() 
                left = int(x - 0.5 * w)
                top = int(y - 0.5 * h)
                width = int(w)
                height = int(h )
                box = np.array([left, top, width, height])
                boxes.append(box)
                
        r_class_ids, r_confs, r_boxes = list(), list(), list()

        indexes = cv2.dnn.NMSBoxes(boxes, confs, 0.15, 0.45) 
        for i in indexes:
            r_class_ids.append(class_ids[i])
            r_confs.append(confs[i])
            r_boxes.append(boxes[i])
        
        for i, class_id in enumerate(r_class_ids):
            left, top, width, height = r_boxes[i]
            conf = r_confs[i]
            cv2.rectangle(self.input, (left, top), (left + width, top + height), (0,255,0), 3)
            cv2.putText(self.input, self.names[r_class_ids[i]], (left + 10, top + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

        return self.input

class cv2onnxModel(onnxModel):
    def __init__(self, name_: str):
        super().__init__(name_)
        self.model = cv2.dnn.readNetFromONNX(f'{self.name}.onnx')
    def inference(self, input_):
        self.input = input_
        blob = cv2.dnn.blobFromImage(
            input_, 
            1/255.0, 
            input_.shape[0:2], 
            swapRB=True, crop=False
        )
        self.model.setInput(blob)
        self.output = self.model.forward()
        self.output = self.output.transpose((0, 2, 1))
        return self
    pass

class onnxVanillaModel(onnxModel):
    def __init__(self, name_: str):
        super().__init__(name_)
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.model = ort.InferenceSession(f'{self.name}.onnx', providers=['MIGraphXExecutionProvider', ], sess_options=so)
    def inference(self, input_):
        self.input = input_
        blob = cv2.dnn.blobFromImage(
            input_, 
            1/255.0, 
            input_.shape[0:2], 
            swapRB=True, crop=False
        )
        self.output = np.array(
            self.model.run(
                output_names='', 
                input_feed={'images': blob})[0]).transpose((0,2,1))
        return self

class ptModel(ModelWrapper):
    def __init__(self, name_: str):
        super().__init__(name_)
        self.model = YOLO(f'{self.name}.pt')
    def inference(self, input_):
        self.output = self.ptModel(input_)
        return self
    def drawResult(self):
        pass