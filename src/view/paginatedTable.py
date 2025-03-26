from PyQt6.QtWidgets import QWidget, QVBoxLayout, QTableWidget, QPushButton, QLabel, QLineEdit, QHBoxLayout, QTableWidgetItem, QHeaderView, QTextEdit
from PyQt6.QtCore import pyqtSignal, Qt

class PaginatedTableWidget(QWidget):
    closeRequested = pyqtSignal()
    def __init__(self, data, rows_per_page=10):
        """
        data: list of tuples, each tuple should be (weight label, weight value)
        rows_per_page: number of rows to show per page
        """
        super().__init__()
        self.data = data
        self.rows_per_page = rows_per_page
        self.current_page = 1
        self.total_pages = max(1, (len(self.data) + rows_per_page - 1) // rows_per_page)
        self.node_name = ""
        
        self._initUI()


    def _initUI(self):
        
        layout = QVBoxLayout(self)

        # Pagination Controls
        topBarLayout = QHBoxLayout()
        topBarLayout.addStretch()
        self.closeBtn = QPushButton("X")
        self.closeBtn.setStyleSheet("""
            QPushButton {
                background-color: #CD5C5C;
                border: none;
                color: white;
                padding: 4px 6px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #e06666; 
            }
            QPushButton:pressed {
                background-color: #a94442;  
            }
        """)

        self.titleWidgets = QTextEdit()
        self.titleWidgets.setText(self.node_name)
        self.titleWidgets.setReadOnly(True)
        self.titleWidgets.setStyleSheet("""
            QTextEdit {
                font-weight: bold;
                font-size : 16px;
            }
        """)       

        self.closeBtn.clicked.connect(self.closeRequested.emit)
        topBarLayout.addWidget(self.titleWidgets)
        topBarLayout.addWidget(self.closeBtn)
        layout.addLayout(topBarLayout)
        
        self.table = QTableWidget()
        self.table.setColumnCount(3)
        self.table.setHorizontalHeaderLabels(["Label","Gradient", "Weight"])
        layout.addWidget(self.table)
        self.table.setStyleSheet("""
            QTableWidget {
                background-color: #f9f9f9;
                color: #333;
                gridline-color: #ccc;
                font-size: 14px;
                border: 1px solid #ddd;
            }
            QHeaderView::section {
                padding: 6px;
                border: 1px solid #6c6c6c;
            }
            QTableWidget::item:selected {
                background-color: #e9e9e9;
            }
        """)
        

        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)

        # Pagination Control
        self.paginationLayout = QHBoxLayout()
        self.prevButton = QPushButton("Previous")
        self.prevButton.clicked.connect(self._goToPreviousPage)
        self.nextButton = QPushButton("Next")
        self.nextButton.clicked.connect(self._goToNextPage)
        self.pageLabel = QLabel(f"Page {self.current_page} of {self.total_pages}")
        self.pageEdit = QLineEdit()
        self.pageEdit.setFixedWidth(40)
        self.pageEdit.setPlaceholderText("Page")
        self.pageEdit.returnPressed.connect(self._goToPage)
        self.paginationLayout.addWidget(self.prevButton)
        self.paginationLayout.addWidget(self.nextButton)
        self.paginationLayout.addWidget(self.pageLabel)
        self.paginationLayout.addWidget(self.pageEdit)
        layout.addLayout(self.paginationLayout)

        self._loadPage(self.current_page)

    def updateTitle(self):
        self.titleWidgets.setText(self.node_name)
        self.titleWidgets.setFixedHeight(40)
        
    def _loadPage(self, page):
        self.current_page = page
        start_index = (page - 1) * self.rows_per_page
        end_index = start_index + self.rows_per_page
        page_data = self.data[start_index:end_index]
        self.table.setRowCount(len(page_data))
        for row, (label, grads, weight) in enumerate(page_data):
            item_label = QTableWidgetItem(str(label))
            item_label.setFlags(item_label.flags() & ~Qt.ItemFlag.ItemIsEditable)
            grads_value = QTableWidgetItem(str(grads))
            grads_value.setFlags(grads_value.flags() & ~Qt.ItemFlag.ItemIsEditable)
            weight_value = QTableWidgetItem(str(weight))
            weight_value.setFlags(weight_value.flags() & ~Qt.ItemFlag.ItemIsEditable)
            
            self.table.setItem(row, 0, item_label)
            self.table.setItem(row, 1, grads_value)
            self.table.setItem(row, 2, weight_value)
        self.pageLabel.setText(f"Page {self.current_page} of {self.total_pages}")

    def _goToPreviousPage(self):
        if self.current_page > 1:
            self._loadPage(self.current_page - 1)

    def _goToNextPage(self):
        if self.current_page < self.total_pages:
            self._loadPage(self.current_page + 1)

    def _goToPage(self):
        try:
            page = int(self.pageEdit.text())
            if 1 <= page <= self.total_pages:
                self._loadPage(page)
        except ValueError:
            pass

    def updateData(self, new_data, color, node_name):
        """Updates the table data and resets the pagination."""
        self.node_name = node_name
        self.updateTitle()
        self.data = new_data
        self.color = color
        self.current_page = 1
        self.total_pages = max(1, (len(self.data) + self.rows_per_page - 1) // self.rows_per_page)
        self._loadPage(self.current_page)
        self.table.setStyleSheet(f"""
            QHeaderView::section {{
                background-color: {self.color};
            }}
        """)
