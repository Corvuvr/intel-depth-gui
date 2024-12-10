import sys
import cv2
import os
import copy
import numpy as np

from PyQt6.QtWidgets import QApplication, QMainWindow, QLabel, QLineEdit, QVBoxLayout, QHBoxLayout, QWidget, QPushButton, QFileDialog, QSizePolicy
from PyQt6.QtGui import QIcon, QPixmap, QImage
from PyQt6.QtCore import Qt,  pyqtSignal, QThread, QRect, QPoint

from camera import MonoCamera, DepthCamera

cam = DepthCamera()
global tmp

def getPointCoordinates(depth_):
    return (0, 0, 0)

class ThreadOpenCV(QThread):
    signalChangePixmap = pyqtSignal(QImage)
    def __init__(self, app_: QLabel):
        super().__init__()
        self.isRunning = False
        self.app = app_
        self.lastCursorPos = (0,0)
    def run(self):
        self.isRunning = True
        while self.isRunning:
            frame, depth = cam.readBuffer()
            h,w,_ = frame.shape # 480 640
            scaledH, scaledW = self.app.videoBufferHeight, self.app.videoBufferWidth # 720 960
            yFactor = h / scaledH
            xFactor = w / scaledW
            frame = self.drawDepth(
                inputRGB_=frame, 
                inputDepth_=depth, 
                cursorCoordinates_= tmp,
                scaleFactor_=(xFactor, yFactor)
            )
            convertToQtFormat = QImage(frame.data, w, h, QImage.Format.Format_RGB888) 
            p = convertToQtFormat.scaled(960,720,Qt.AspectRatioMode.KeepAspectRatio)
            self.signalChangePixmap.emit(p)                    
            self.msleep(20)      
    def stop(self):
        self.isRunning = False
        self.quit()     

    def drawDepth(self, inputRGB_, inputDepth_, cursorCoordinates_, scaleFactor_):
        x, y, z = getPointCoordinates(depth_=inputDepth_)
        #print(inputRGB_.shape)
        # inputRGB_ = cv2.resize(inputRGB_, ())
        xCenter =  int(cursorCoordinates_[0]*scaleFactor_[0])
        yCenter =  int(cursorCoordinates_[1]*scaleFactor_[1])
        d= inputDepth_[yCenter][xCenter]
        cv2.rectangle(
            inputRGB_, 
            (xCenter + 10, yCenter + 10), 
            (xCenter - 10, yCenter - 10), 
            (0,255,0), 
            1
        )
        cv2.putText(inputRGB_, f'{d / 1000 }m', (xCenter + 20, yCenter + 7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        return inputRGB_
class ClickWidget(QLabel):
    pressPos = None
    clicked = pyqtSignal()
    def __init__(self):
        super().__init__()
        self.lastClickGlobalPos = (0,0)
    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.pressPos = QPoint(event.pos())
            #self.lastClickGlobalPos = tuple((self.pressPos.x(), self.pressPos.y()))
            self.lastClickGlobalPos = tuple((
                QPoint(event.pos()).x(), QPoint(event.pos()).y()))
            global tmp 
            tmp = copy.copy(self.lastClickGlobalPos)
            
    def mouseReleaseEvent(self, event):
        # ensure that the left button was pressed *and* released within the
        # geometry of the widget; if so, emit the signal;
        if (self.pressPos is not None and 
            event.button() == Qt.MouseButton.LeftButton and 
            event.pos() in self.rect()):
                self.clicked.emit()
        self.pressPos = None

class MainWindow(QMainWindow):
    def __init__(self, screenSize_: tuple[int], assetPath_: str):
        super(MainWindow, self).__init__()
        self.assetPath = f'{assetPath_}'
        global tmp
        tmp = (0,0)
        # Dimensions
        VIDEOASPECTRATIO_W, VIDEOASPECTRATIO_H = 4, 3
        self.screenH, self.screenW = screenSize_
        self.winH, self.winW = (int(x/2) for x in screenSize_)
        self.videoBufferHeight, self.videoBufferWidth = \
            self.winH, int(self.winH / VIDEOASPECTRATIO_H * VIDEOASPECTRATIO_W)
        self.sidebarWidth = self.winW - self.videoBufferWidth
        # Main Window
        self.setWindowTitle("Legosearch")
        self.setGeometry(
            int((self.screenH - self.winH) / 2), 
            int((self.screenW - self.winW) / 2), 
            self.winW, 
            self.winH)
        self.setFixedHeight(self.winH)
        self.setFixedWidth(self.winW)
        self.setStyleSheet("background-color: #474747;")
        # Layouts
        layoutMainVertical = QHBoxLayout()
        layoutMainVertical.setGeometry(QRect(0, 0, self.winW, self.winH))
        layoutMainVertical.setAlignment(Qt.AlignmentFlag.AlignTop)
        layoutMainVertical.setContentsMargins(0,0,0,0)
        layoutSidebar = QVBoxLayout()
        layoutSidebar.setGeometry(QRect(0, 0, self.sidebarWidth, self.winH))
        layoutSidebar.setAlignment(Qt.AlignmentFlag.AlignTop)
        
        # Video Buffer
        self.lastClickGlobalPos = ...
        self.addVideo(layout_=layoutMainVertical)
        # Search Image
        self.addPhoto(layout_=layoutSidebar)
        # BUTTONS
        self.buttonCount = 0
        self.styleSheet_buttons = self.getBtnStyle()
        # Open Button
        self.addButton("OpenImage", self.event_openfile, layoutSidebar)
        # Run Button
        self.addButton("Camera: off", self.switchCam, layoutSidebar)
        print(self.pos())
        # Bring To Life
        layoutMainVertical.addLayout(layoutSidebar)
        widget = QWidget()
        widget.setLayout(layoutMainVertical)
        self.setCentralWidget(widget)
        self.thread = ThreadOpenCV(self)
        self.thread.signalChangePixmap.connect(self.setImage)

    def addVideo(self, layout_):
        self.labelVideoBuffer = ClickWidget()
        self.labelVideoBuffer.setPixmap(self.convertCV2Qt(cv2.imread(f'{self.assetPath}/offline.png')))
        self.labelVideoBuffer.setFixedHeight(self.videoBufferHeight)
        self.labelVideoBuffer.setFixedWidth(self.videoBufferWidth)
        layout_.addWidget(self.labelVideoBuffer)
    def addPhoto(self, layout_):
        self.labelSearchImage = QLabel("Photo")
        self.labelSearchImage.setGeometry(self.videoBufferWidth, 0, self.sidebarWidth, self.sidebarWidth)
        self.labelSearchImage.setPixmap(self.convertCV2Qt(cv2.imread(f'{self.assetPath}/imagePlaceholder.jpg')))
        layout_.addWidget(self.labelSearchImage)

    def addButton(self, name_, function_, layout_):
        self.buttonCount += 1
        buttonHeight = int((self.winH - self.sidebarWidth) / self.buttonCount)
        self.newButton = QPushButton(f'{name_}')
        self.newButton.setGeometry(0, int(self.sidebarWidth + buttonHeight * 0), self.sidebarWidth, buttonHeight)
        self.newButton.setStyleSheet(self.styleSheet_buttons)
        self.newButton.clicked.connect(lambda: function_(self.newButton))
        layout_.addWidget(self.newButton)

    def event_openfile(self):
        res = QFileDialog.getOpenFileName(self, 'Открытие файла', os.getcwd(),'Images (*.png *.jpg)')
        self.labelSearchImage.setPixmap(QPixmap(res[0]))
        if len(res[0])==0:
            self.labelSearchImage.setPixmap(QPixmap(f"{self.assetPath}/Detail_photo.png"))
        print(f'{"-"*33}\n{res}')
         
    def switchCam(self, btn_):
        if btn_.text() == "Camera: off":
            self.thread.start()
            btn_.setText("Camera: on")
        else:
            self.thread.stop()
            btn_.setText("Camera: off")
            self.labelVideoBuffer.setPixmap(self.convertCV2Qt(cv2.imread(f'{self.assetPath}/offline.png')))
           
    def setImage(self,image):
        self.labelVideoBuffer.setPixmap(QPixmap.fromImage(image))
    
    def convertCV2Qt(self, cvImg):
        """Convert from an opencv image to QPixmap"""
        rgbImage = cv2.cvtColor(cvImg, cv2.COLOR_BGR2RGB)
        h, w, ch = rgbImage.shape
        bytesPerLine = ch * w
        qImage = QImage(rgbImage.data, w, h, bytesPerLine, QImage.Format.Format_RGB888)
        return QPixmap.fromImage(qImage)
    
    def getBtnStyle(self) -> str:
        return """
            QPushButton {
                font-family: Consolas;
                font-size: 20px;
                background-color: #5B5B5B; /* фон */
                color: white;             /* Белый текст */
                border: none; 
                padding: 10;
            }
            QPushButton:hover {
                background-color: #808080; /* Цвет при наведении */
            }
        """
def createApp():
    app = QApplication(sys.argv)
    appSize = (
        app.primaryScreen().size().height(),
        app.primaryScreen().size().width(),
    )
    window = MainWindow(screenSize_ = appSize, assetPath_='assets')
    window.show()
    app.exec()

if __name__ == "__main__":
    createApp()