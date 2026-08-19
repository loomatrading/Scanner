import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QHBoxLayout, 
                             QVBoxLayout, QPushButton, QLabel, QListWidget, 
                             QListWidgetItem, QCheckBox, QFileDialog, QFrame, QSplitter)
from PyQt5.QtGui import QImage, QPixmap, QIcon, QFont
from PyQt5.QtCore import Qt, QSize

# ==========================================
# 1. خوارزميات معالجة الصور والذكاء الاصطناعي
# ==========================================

class ImageProcessor:
    @staticmethod
    def auto_detect_and_crop(image):
        """كشف زوايا المستند تلقائياً وتعديل المنظور"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edged = cv2.Canny(blur, 75, 200)

        # البحث عن الحدود
        cnts, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]

        doc_cnt = None
        for c in cnts:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            if len(approx) == 4:
                doc_cnt = approx
                break

        if doc_cnt is not None:
            return ImageProcessor.four_point_transform(image, doc_cnt.reshape(4, 2))
        return image

    @staticmethod
    def order_points(pts):
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        return rect

    @staticmethod
    def four_point_transform(image, pts):
        rect = ImageProcessor.order_points(pts)
        (tl, tr, br, bl) = rect

        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        maxWidth = max(int(widthA), int(widthB))

        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        maxHeight = max(int(heightA), int(heightB))

        dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]], dtype="float32")

        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
        return warped

    @staticmethod
    def apply_magic_pro(image):
        """فليتر Magic Pro لإزالة الظلال والتجاعيد وتبييض الخلفية"""
        # تحويل الصورة لتنظيف الظلال الكبيرة (Shadow Removal)
        rgb_planes = cv2.split(image)
        result_norm_planes = []
        for plane in rgb_planes:
            dilated_img = cv2.dilate(plane, np.ones((7,7), np.uint8))
            bg_img = cv2.medianBlur(dilated_img, 21)
            diff_img = 255 - cv2.absdiff(plane, bg_img)
            norm_img = cv2.normalize(diff_img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
            result_norm_planes.append(norm_img)
        
        cleaned = cv2.merge(result_norm_planes)
        
        # تحسين حدة النص وزيادة التباين
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        sharpened = cv2.filter2D(cleaned, -1, kernel)
        return sharpened

    @staticmethod
    def apply_enhance(image):
        """تحسين التباين والألوان"""
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl,a,b))
        return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    @staticmethod
    def apply_lighten(image):
        """تفتيح الخلفية"""
        return cv2.convertScaleAbs(image, alpha=1.2, beta=30)

    @staticmethod
    def apply_bw(image):
        """أسود وأبيض ناصع للمستندات"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 10)


# ==========================================
# 2. تصميم الواجهة الرئيسية (PyQt5)
# ==========================================

class CamScannerUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("CamScanner PC Studio")
        self.setGeometry(100, 100, 1200, 700)
        self.setStyleSheet("background-color: #F4F5F7;")

        self.original_image = None
        self.processed_image = None

        self.init_ui()

    def init_ui(self):
        main_layout = QVBoxLayout()
        
        # 1. Header Area (Top)
        top_bar = QHBoxLayout()
        btn_import = QPushButton("  Import Images")
        btn_import.setStyleSheet("""
            QPushButton {
                background-color: #FFFFFF;
                border: 1px solid #DCDFE6;
                border-radius: 8px;
                padding: 8px 20px;
                font-size: 14px;
                font-weight: bold;
                color: #303133;
            }
            QPushButton:hover { background-color: #F2F6FC; }
        """)
        btn_import.clicked.connect(self.import_image)
        top_bar.addStretch()
        top_bar.addWidget(btn_import)
        top_bar.addStretch()
        main_layout.addLayout(top_bar)

        # 2. Middle Area (Splitter for Left Thumbnails, Center View, Right Controls)
        content_layout = QHBoxLayout()

        # Left Panel (Thumbnails)
        left_panel = QVBoxLayout()
        self.thumb_list = QListWidget()
        self.thumb_list.setFixedWidth(120)
        self.thumb_list.setStyleSheet("background-color: #FAFAFA; border: 1px solid #E4E7ED;")
        left_panel.addWidget(self.thumb_list)

        # Center Panel (Main Image Canvas)
        self.image_display = QLabel("No Image Loaded")
        self.image_display.setAlignment(Qt.AlignCenter)
        self.image_display.setStyleSheet("background-color: #EBF0F5; border: 1px solid #DCDFE6; border-radius: 4px;")

        # Right Panel (Tools & Enhancements)
        right_panel = QVBoxLayout()
        right_panel.setContentsMargins(10, 0, 10, 0)

        # Transform Tools (Rotate, Flip, Crop)
        tool_grid = QHBoxLayout()
        btn_rotate_l = QPushButton("⟲")
        btn_rotate_r = QPushButton("2")
        btn_crop = QPushButton("✂ Crop")
        
        for btn in [btn_rotate_l, btn_rotate_r, btn_crop]:
            btn.setStyleSheet("background-color: #FFFFFF; border: 1px solid #DCDFE6; padding: 8px; border-radius: 4px;")
        
        btn_crop.clicked.connect(self.handle_crop)

        tool_grid.addWidget(btn_rotate_l)
        tool_grid.addWidget(btn_rotate_r)
        tool_grid.addWidget(btn_crop)
        right_panel.addLayout(tool_grid)

        # Filter Options Buttons
        right_panel.addSpacing(20)
        
        self.btn_original = QPushButton("Original")
        self.btn_enhance = QPushButton("Enhance")
        self.btn_magic = QPushButton(" Magic Pro")
        self.btn_lighten = QPushButton("Lighten")
        self.btn_bw = QPushButton("B&W")

        # Style Magic Pro distinctly
        self.btn_magic.setStyleSheet("""
            QPushButton {
                background-color: #E6F7F5;
                color: #00A896;
                border: 2px solid #00A896;
                font-weight: bold;
                border-radius: 8px;
                padding: 12px;
            }
        """)

        filter_buttons = [self.btn_original, self.btn_enhance, self.btn_lighten, self.btn_bw]
        for btn in filter_buttons:
            btn.setStyleSheet("""
                QPushButton {
                    background-color: #FFFFFF;
                    border: 1px solid #DCDFE6;
                    border-radius: 8px;
                    padding: 12px;
                    font-size: 13px;
                }
                QPushButton:hover { background-color: #F5F7FA; }
            """)

        right_panel.addWidget(self.btn_original)
        right_panel.addWidget(self.btn_enhance)
        right_panel.addWidget(self.btn_magic)
        right_panel.addWidget(self.btn_lighten)
        right_panel.addWidget(self.btn_bw)

        # Connections
        self.btn_original.clicked.connect(lambda: self.apply_filter("original"))
        self.btn_enhance.clicked.connect(lambda: self.apply_filter("enhance"))
        self.btn_magic.clicked.connect(lambda: self.apply_filter("magic"))
        self.btn_lighten.clicked.connect(lambda: self.apply_filter("lighten"))
        self.btn_bw.clicked.connect(lambda: self.apply_filter("bw"))

        # Checkbox & Confirm
        right_panel.addStretch()
        chk_apply_all = QCheckBox("Apply to All Pages")
        btn_confirm = QPushButton("Confirm")
        btn_confirm.setStyleSheet("""
            QPushButton {
                background-color: #00B094;
                color: white;
                font-size: 15px;
                font-weight: bold;
                border-radius: 6px;
                padding: 10px;
            }
            QPushButton:hover { background-color: #008F78; }
        """)

        right_panel.addWidget(chk_apply_all)
        right_panel.addWidget(btn_confirm)

        # Assemble Panels
        content_layout.addLayout(left_panel, 1)
        content_layout.addWidget(self.image_display, 5)
        content_layout.addLayout(right_panel, 2)

        main_layout.addLayout(content_layout)

        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

    # ==========================================
    # 3. الوظائف والتفاعل
    # ==========================================

    def import_image(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Document Image", "", "Image Files (*.png *.jpg *.jpeg *.bmp)")
        if file_path:
            # قراءة الصورة
            img = cv2.imread(file_path)
            
            # 1. ضبط الزوايا والقص تلقائياً فور فتح الملف
            cropped_img = ImageProcessor.auto_detect_and_crop(img)
            self.original_image = cropped_img.copy()
            
            # 2. تطبيق فلتر Magic Pro تلقائياً للتحسين الافتراضي
            self.apply_filter("magic")
            
            # إضافة الصورة للجانب الأيسر
            item = QListWidgetItem("Page 1")
            self.thumb_list.addItem(item)

    def handle_crop(self):
        """إعادة فتح الصورة والأطراف لضبط الزوايا يدوياً"""
        if self.original_image is not None:
            # يمكن ربطه بمرسم النقاط الأربع لضبط الحدود يدوياً
            cropped = ImageProcessor.auto_detect_and_crop(self.original_image)
            self.original_image = cropped
            self.apply_filter("magic")

    def apply_filter(self, filter_type):
        if self.original_image is None:
            return

        if filter_type == "original":
            self.processed_image = self.original_image.copy()
        elif filter_type == "enhance":
            self.processed_image = ImageProcessor.apply_enhance(self.original_image)
        elif filter_type == "magic":
            self.processed_image = ImageProcessor.apply_magic_pro(self.original_image)
        elif filter_type == "lighten":
            self.processed_image = ImageProcessor.apply_lighten(self.original_image)
        elif filter_type == "bw":
            self.processed_image = ImageProcessor.apply_bw(self.original_image)

        self.display_image(self.processed_image)

    def display_image(self, img):
        """عرض الصورة المعالجة داخل واجهة البرنامج"""
        if len(img.shape) == 2:
            h, w = img.shape
            q_img = QImage(img.data, w, h, w, QImage.Format_Grayscale8)
        else:
            h, w, c = img.shape
            bytes_per_line = 3 * w
            rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            q_img = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)

        pixmap = QPixmap.fromImage(q_img)
        scaled_pixmap = pixmap.scaled(self.image_display.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.image_display.setPixmap(scaled_pixmap)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = CamScannerUI()
    window.show()
    sys.exit(app.exec_())
