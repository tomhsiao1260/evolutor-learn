import sys
from PyQt5.QtWidgets import (
        QApplication,
        QWidget,
        QPushButton,
        QVBoxLayout,
        QMessageBox,
        )

def on_button_click():
    QMessageBox.information(window, "message", "button click!")

# python wind2d.py
if __name__ == '__main__':
    app = QApplication(sys.argv)

    window = QWidget()
    window.setWindowTitle("Simple QApplication Example")

    layout = QVBoxLayout()

    button = QPushButton("click me")
    button.clicked.connect(on_button_click)

    layout.addWidget(button)
    window.setLayout(layout)

    window.show()
    sys.exit(app.exec())