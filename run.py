import sys
from PyQt5.QtWidgets import *
if __name__ == "__main__":
    app = QApplication(sys.argv)
    vc = QLabel("hello world")
    hl = QHBoxLayout()
    hl.addWidget(vc)
    w = QWidget()
    w.setLayout(hl)
    w.resize(300,300)
    w.show()
    sys.exit(app.exec_())