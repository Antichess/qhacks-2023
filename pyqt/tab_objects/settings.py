import sys
from PyQt5.QtWidgets import *
from PyQt5 import QtCore, QtWidgets
from tab_objects.camera_test import App

class settings(QWidget):
    def __init__(self):
        
        self.labels = [
            ["Save logs","checkbox"],
            ["Test","checkbox"]
        ]

        pass
    
    def layout(self):
        self.control_objects = [] # this saves pointers to checkboxes, qlineedits, etc
        self.label_objects = [] # this saves pointers to labels
        for x in self.labels:
            if x[1] == "checkbox":
                self.control_objects.append(QCheckBox())
            self.label_objects.append(QLabel(x[0]))
        
        self.horizontal_rows = [QHBoxLayout() for x in range(len(self.control_objects))] # QHbox layouts
        for c,x in enumerate(self.horizontal_rows):
            x.addWidget(self.label_objects[c],1)
            x.addWidget(self.control_objects[c],2)

        self.horizontal_widgets = [QWidget() for x in range(len(self.control_objects))]
        [self.horizontal_widgets[c].setLayout(x) for c,x in enumerate(self.horizontal_rows)]

        self.vertical_stack = QVBoxLayout()
        self.vertical_stack.setAlignment(QtCore.Qt.AlignTop)
        [self.vertical_stack.addWidget(x) for x in self.horizontal_widgets]
        self.vertical_widget = QWidget()
        self.vertical_widget.setLayout(self.vertical_stack)


        
        return self.vertical_widget
