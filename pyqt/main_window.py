import sys
from PyQt5.QtWidgets import *
from camera_test import App

class mainObj(QWidget):
    def __init__(self):
        pass
    
    def layout(self):
        self.tab_layout = QTabWidget()

        #TAB 1
        self.tab1_layout = QVBoxLayout()
        self.tab1_layout.addWidget(QLabel("Hello World!"))

        self.tab1 = QWidget()
        self.tab1.setLayout(self.tab1_layout)

        #TAB 2
        self.tab2_layout = QVBoxLayout()
        self.camera = App()
        self.tab2_layout.addWidget(self.camera)

        self.tab2 = QWidget()
        self.tab2.setLayout(self.tab2_layout)

        self.tab_layout.addTab(self.tab1, "Text")
        self.tab_layout.addTab(self.tab2, "Camera")
        
        self.final_qv_tab_layout = QVBoxLayout()
        self.final_qv_tab_layout.addWidget(self.tab_layout)

        self.w = QWidget()
        self.w.setLayout(self.final_qv_tab_layout)
        
        return self.w
