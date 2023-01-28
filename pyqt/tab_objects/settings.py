import sys
from PyQt5.QtWidgets import *
from PyQt5 import QtCore, QtWidgets
from tab_objects.camera_test import App

class settings(QWidget):
    def __init__(self):
        #super().__init__()
        
        self.labels = [
            ["Save logs","checkbox"],
            ["Test","qlineedit"]
        ]

        self.control_objects = [] # this saves pointers to checkboxes, qlineedits, etc
        self.label_objects = [] # this saves pointers to labels

        pass
    
    def layout(self):
        
        for x in self.labels:
            self.control_objects.append(QCheckBox()) if x[1] == "checkbox" else False
            self.control_objects.append(QLineEdit()) if x[1] == "qlineedit" else False
                
            self.label_objects.append(QLabel(x[0]))
        
        self.control_objects[0].stateChanged.connect(lambda:self.save_logs(self.control_objects[0]))

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

    def save_logs(self,b):
        if b.isChecked():
            print("checked")
        else:
            print("unchecked")

    def test(self):
        pass
