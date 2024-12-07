# main.py

import sys
from PySide6.QtWidgets import QApplication
from balance import BalanceWindow

def main():
    app = QApplication(sys.argv)
    window = BalanceWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
