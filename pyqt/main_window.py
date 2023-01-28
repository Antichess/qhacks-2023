import sys
from PyQt5.QtWidgets import *
from camera_test import App

class mainObj(QWidget):
    def __init__(self):
        pass
    
    def layout(self):
        self.label = QLabel("Hello world!")
        self.w_layout = QVBoxLayout()
        self.w_layout.addWidget(self.label)

        self.w = QWidget()
        self.w.setLayout(self.w_layout)
        
        return self.w
