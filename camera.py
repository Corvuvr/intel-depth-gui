from abc import ABC, abstractmethod

import cv2
import numpy as np

import pyrealsense2 as rs

class CameraWrapper(ABC):
    def __init__(self):
        self.capture = None
    @abstractmethod
    def readBuffer(self):
        pass

class MonoCamera(CameraWrapper):
    def __init__(self):
        super().__init__()
        self.capture = cv2.VideoCapture(0)
    def readBuffer(self):
        _, frame = self.capture.read()
        return frame

class DepthCamera(CameraWrapper):
    def __init__(self):
        super().__init__()
        config = rs.config()
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        self.capture = rs.pipeline()
        self.capture.start(config)
    def readBuffer(self):
        frame = self.capture.wait_for_frames().get_depth_frame()
        depth_image = np.asanyarray(frame.get_data())
        # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
        return depth_colormap, depth_image