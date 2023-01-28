import sys
import os
from PyQt5.QtWidgets import *
from PyQt5 import QtCore, QtWidgets
from tab_objects.camera_test import App
from tab_objects.settings import settings
from tab_objects.main_menu import menu

class mainObj(QWidget):
    def __init__(self):
        pass
    
    def layout(self):
        self.tab_layout = QTabWidget()

        #TAB 1
        self.m = menu()
        self.main_menu = self.m.layout()

        #TAB 2
        self.camera_tab_layout = QVBoxLayout()
        self.camera = App()
        self.camera_tab_layout.addWidget(self.camera)
        self.run_button = QPushButton("Take Picture")
        self.run_button.clicked.connect(self.camera.thread.take_image)
        self.camera_tab_layout.addWidget(self.run_button)

        self.camera_tab = QWidget()
        self.camera_tab.setLayout(self.camera_tab_layout)

        #TAB 3
        self.settings = settings()
        self.settings_layout = self.settings.layout()
        
        self.tab_layout.addTab(self.camera_tab, "Camera")
        self.tab_layout.addTab(self.main_menu, "Main Menu")
        self.tab_layout.addTab(self.settings_layout, "Settings")
        
        self.final_qv_tab_layout = QVBoxLayout()
        self.final_qv_tab_layout.addWidget(self.tab_layout)

        self.w = QWidget()
        self.w.setLayout(self.final_qv_tab_layout)

        self.read_settings()
        
        return self.w

    def read_settings(self):
        with open(os.path.join(os.getcwd(), "tab_objects", "settings.txt")) as f:
            a = f.readlines()
            l = []
            for x in a:
                l.append(x.replace("\n","").split(" ")[1])
            print(l)


        
        pass