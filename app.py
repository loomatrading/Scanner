import sys
import cv2
import numpy as np
import os
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QHBoxLayout, 
                             QVBoxLayout, QPushButton, QLabel, QListWidget, 
                             QListWidgetItem, QCheckBox, QFileDialog, QFrame, 
                             QSplitter, QSizePolicy, QDialog)
from PyQt5.QtGui import QImage, QPixmap, QIcon, QFont, QColor
from PyQt5.QtCore import Qt, QSize, QPoint

# ==========================================
# 1. محرك معالجة الصور الاحترافي (Deep Enhancement Engine)
# ==========================================

class ImageProcessor Pro:
    @staticmethod
    def advanced_shadow_removal(image):
        """خوارزمية لإزالة الظلال والكرمشة بالكامل وتنظيف الخلفية"""
        # تقييم التباين الأولي
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)

        # تحسين التباين التكيفي (CLAHE)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        cl = clahe.apply(l_channel)

        # دمج القنوات مرة أخرى
        enhanced_lab = cv2.merge((cl, a_channel, b_channel))
        cleaned_color = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

        # محاكاة تأثير Magic Pro: تحويل لتوضيح النص مع خلفية نظيفة
        gray = cv2.cvtColor(cleaned_color, cv2.COLOR_BGR2GRAY)
        final = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv2.THRESH_BINARY, 11, 2)
        
        # تحويل النتيجة لتبدو كصورة ملونة ولكن بنص أسود نقي وخلفية بيضاء
        return cv2.cvtColor(final, cv2.COLOR_GRAY2BGR)

    @staticmethod
    def auto_detect_and_crop(image):
        """كشف زوايا المستند تلقائياً وتعديل المنظور (تحسين الدقة)"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
         edged = cv2.Canny(blur, 50, 150, apertureSize=3)

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
            return ImageProcessorPro.four_point_transform(image, doc_cnt.reshape(4, 2))
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
        rect = ImageProcessorPro.order_points(pts)
        (tl, tr, br, bl) = rect

        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        maxWidth = max(int(widthA), int(widthB))

        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - br[0]) ** 2) + ((tl[1] - br[1]) ** 2))
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
    def rotate_image(image, angle):
        """تدوير الصورة بزاوية محددة"""
        (h, w) = image.shape[:2]
        (cX, cY) = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])
        nW = int((h * sin) + (w * cos))
        nH = int((h * cos) + (w * sin))
        M[0, 2] += (nW / 2) - cX
        M[1, 2] += (nH / 2) - cY
        return cv2.warpAffine(image, M, (nW, nH))

    @staticmethod
    def apply_enhance(image):
        return cv2.convertScaleAbs(image, alpha=1.3, beta=20)

    @staticmethod
    def apply_lighten(image):
        return cv2.convertScaleAbs(image, alpha=1.1, beta=50)

    @staticmethod
    def apply_bw(image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return cv2.cvtColor(cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1], cv2.COLOR_GRAY2BGR)


# ==========================================
# 2. تصميم الواجهة الاحترافية (Pro UI)
# ==========================================

class CamScannerPro(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("CamScanner PC Studio Pro")
        self.setGeometry(100, 100, 1200, 750)
        self.setAcceptDrops(True)
        self.original_image = None
        self.processed_image = None
        self.filter_applied = "magic" # Default filter

        self.init_ui()

    def init_ui(self):
        # تطبيق التخصيص العام (QSS)
        self.setStyleSheet("""
            QMainWindow { background-color: #F8F9FB; }
            QWidget { font-family: 'Segoe UI', Arial; font-size: 13px; }
            QPushButton { border-radius: 6px; font-weight: bold; }
        """)

        central_widget = QWidget()
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(15, 10, 15, 15)

        # 1. Header Area
        top_bar = QHBoxLayout()
        top_bar.addStretch()
        self.btn_import = QPushButton("  Import Images")
        self.btn_import.setStyleSheet("""
            QPushButton {
                background-color: #FFFFFF;
                border: 1px solid #DCDFE6;
                padding: 10px 25px;
                font-size: 14px;
                color: #303133;
            }
            QPushButton:hover { background-color: #F5F7FA; border-color: #C0C4CC; }
        """)
        self.btn_import.clicked.connect(self.import_image)
        top_bar.addWidget(self.btn_import)
        top_bar.addStretch()
        main_layout.addLayout(top_bar)

        # 2. Main Content (Splitter)
        content_splitter = QSplitter(Qt.Horizontal)
        content_splitter.setStyleSheet("QSplitter::handle { background-color: #EBF0F5; width: 4px; }")

        # Left Panel (Thumbnails)
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        self.thumb_list = QListWidget()
        self.thumb_list.setStyleSheet("""
            QListWidget {
                background-color: #FFFFFF;
                border: 1px solid #E4E7ED;
                border-radius: 4px;
            }
            QListWidget::item { padding: 10px; border-bottom: 1px solid #F0F2F5; }
            QListWidget::item:selected { background-color: #F0F9F8; color: #00A896; font-weight: bold; }
        """)
        left_layout.addWidget(self.thumb_list)
        content_splitter.addWidget(left_panel)
        content_splitter.setStretchFactor(0, 1)

        # Center Panel (Main View)
        center_panel = QWidget()
        center_layout = QVBoxLayout(center_panel)
        center_layout.setContentsMargins(10, 0, 10, 0)
        self.image_display = QLabel()
        self.image_display.setAlignment(Qt.AlignCenter)
        self.image_display.setMinimumSize(400, 400)
        self.image_display.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.image_display.setStyleSheet("""
            QLabel {
                background-color: #FFFFFF;
                border: 1px solid #E4E7ED;
                border-radius: 4px;
                margin: 5px;
            }
        """)
        center_layout.addWidget(self.image_display)
        content_splitter.addWidget(center_panel)
        content_splitter.setStretchFactor(1, 5)

        # Right Panel (Tools & Enhancements)
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(5, 0, 0, 0)

        # Transform Tools (Icons)
        tools_layout = QHBoxLayout()
        tools_layout.setSpacing(10)
        self.btn_rotate_l = QPushButton("⟲")
        self.btn_rotate_r = QPushButton("2")
        self.btn_crop = QPushButton("✂ Crop")
        
        tool_btn_style = """
            QPushButton {
                background-color: #FFFFFF;
                border: 1px solid #DCDFE6;
                padding: 10px;
                font-size: 16px;
                min-width: 40px;
            }
            QPushButton:hover { background-color: #F5F7FA; }
        """
        self.btn_crop.setStyleSheet(tool_btn_style + "font-size: 13px;")
        for btn in [self.btn_rotate_l, self.btn_rotate_r]:
            btn.setStyleSheet(tool_btn_style)
        
        # Connections for tools
        self.btn_rotate_l.clicked.connect(lambda: self.apply_rotation(-90))
        self.btn_rotate_r.clicked.connect(lambda: self.apply_rotation(90))
        self.btn_crop.clicked.connect(self.handle_manual_crop)

        tools_layout.addWidget(self.btn_rotate_l)
        tools_layout.addWidget(self.btn_rotate_r)
        tools_layout.addWidget(self.btn_crop)
        right_layout.addLayout(tools_layout)

        # Filters buttons
        right_layout.addSpacing(15)
        
        self.filters_btn_style = """
            QPushButton {
                background-color: #FFFFFF;
                border: 1px solid #DCDFE6;
                padding: 15px;
                text-align: left;
            }
            QPushButton:hover { background-color: #F5F7FA; border-color: #C0C4CC; }
            QPushButton:checked {
                background-color: #F0F9F8;
                color: #00A896;
                border: 1px solid #00A896;
                font-weight: bold;
            }
        """

        self.btn_original = QPushButton("Original")
        self.btn_enhance = QPushButton("Enhance")
        self.btn_magic = QPushButton(" Magic Pro")
        self.btn_lighten = QPushButton("Lighten")
        self.btn_bw = QPushButton("BW")

        for btn in [self.btn_original, self.btn_enhance, self.btn_magic, self.btn_lighten, self.btn_bw]:
            btn.setStyleSheet(self.filters_btn_style)
            btn.setCheckable(True)

        self.btn_magic.setChecked(True) # Magic pro is default
        self.btn_magic.setStyleSheet("""
            QPushButton {
                background-color: #EBF8F7;
                color: #00A896;
                border: 1px solid #00A896;
                font-weight: bold;
                padding: 15px;
                text-align: left;
            }
        """)

        right_layout.addWidget(self.btn_original)
        right_layout.addWidget(self.btn_enhance)
        right_layout.addWidget(self.btn_magic)
        right_layout.addWidget(self.btn_lighten)
        right_layout.addWidget(self.btn_bw)

        # Hook filter buttons to action
        self.btn_original.clicked.connect(lambda: self.apply_filter("original"))
        self.btn_enhance.clicked.connect(lambda: self.apply_filter("enhance"))
        self.btn_magic.clicked.connect(lambda: self.apply_filter("magic"))
        self.btn_lighten.clicked.connect(lambda: self.apply_filter("lighten"))
        self.btn_bw.clicked.connect(lambda: self.apply_filter("bw"))

        # Checkbox & Save
        right_layout.addStretch()
        chk_apply_all = QCheckBox("Apply to All Pages")
        chk_apply_all.setStyleSheet("color: #606266; margin-top: 20px;")
        
        self.btn_save = QPushButton("Save")
        self.btn_save.setStyleSheet("""
            QPushButton {
                background-color: #00B094;
                color: white;
                font-size: 16px;
                font-weight: bold;
                border-radius: 6px;
                padding: 12px;
                margin-top: 10px;
            }
            QPushButton:hover { background-color: #008F78; }
        """)
        self.btn_save.clicked.connect(self.save_image)

        right_layout.addWidget(chk_apply_all)
        right_layout.addWidget(self.btn_save)

        content_splitter.addWidget(right_panel)
        content_splitter.setStretchFactor(2, 2)
        
        main_layout.addWidget(content_splitter)

        self.setCentralWidget(central_widget)

    # ==========================================
    # 3. الوظائف والتفاعل (Backend & IO)
    # ==========================================

    def import_image(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Document Image", "", "Image Files (*.png *.jpg *.jpeg *.bmp)")
        if file_path:
            # قراءة الصورة بالكامل
            img = cv2.imread(file_path)
            
            # 1. ضبط الزوايا والقص تلقائياً
            self.original_image = ImageProcessorPro.auto_detect_and_crop(img)
            
            # 2. تطبيق فلتر Magic Pro تلقائياً للتحسين الافتراضي
            self.filter_applied = "magic"
            self.processed_image = ImageProcessorPro.advanced_shadow_removal(self.original_image)
            
            # إضافة الصورة للجانب الأيسر
            self.thumb_list.addItem(QListWidgetItem(f"Page {self.thumb_list.count() + 1}"))
            
            # عرض الصورة
            self.display_image(self.processed_image)

    def display_image(self, img):
        if img is None:
            return

        h, w, c = img.shape
        bytes_per_line = 3 * w
        # تحويل BGR (OpenCV) إلى RGB (PyQt)
        rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        q_img = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)

        pixmap = QPixmap.fromImage(q_img)
        # تصغير الصورة لتناسب النافذة مع الحفاظ على الأبعاد
        scaled_pixmap = pixmap.scaled(self.image_display.width(), self.image_display.height(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.image_display.setPixmap(scaled_pixmap)

    def apply_rotation(self, angle):
        if self.original_image is not None:
            self.original_image = ImageProcessorPro.rotate_image(self.original_image, angle)
            # إعادة تطبيق الفلتر الحالي على الصورة المدورة
            self.apply_filter(self.filter_applied)

    def handle_manual_crop(self):
        """تفعيل نافذة القص اليدوي (للمرحلة القادمة: إضافة واجهة نقاط الزوايا)"""
        # في الوقت الحالي، يقوم بإعادة القص تلقائياً لضبط الأبعاد
        if self.original_image is not None:
             self.original_image = ImageProcessorPro.auto_detect_and_crop(self.original_image)
             self.apply_filter(self.filter_applied)

    def apply_filter(self, filter_type):
        if self.original_image is None:
            return

        self.filter_applied = filter_type
        
        # تنفيذ الفلتر
        if filter_type == "original":
            self.processed_image = self.original_image.copy()
        elif filter_type == "enhance":
            self.processed_image = ImageProcessorPro.apply_enhance(self.original_image)
        elif filter_type == "magic":
            self.processed_image = ImageProcessorPro.advanced_shadow_removal(self.original_image)
        elif filter_type == "lighten":
            self.processed_image = ImageProcessorPro.apply_lighten(self.original_image)
        elif filter_type == "bw":
            self.processed_image = ImageProcessorPro.apply_bw(self.original_image)

        # عرض النتيجة
        self.display_image(self.processed_image)

    def save_image(self):
        """حفظ الصورة المعالجة بنجاح"""
        if self.processed_image is None:
            return

        options = QFileDialog.Options()
        default_dir = os.path.join(os.path.expanduser('~'), 'Documents')
        file_path, _ = QFileDialog.getSaveFileName(self, "Save Enhanced Document", default_dir, "JPEG Image (*.jpg);;PNG Image (*.png);;All Files (*)", options=options)
        
        if file_path:
            # التأكد من الامتداد الصحيح
            if not (file_path.lower().endswith('.jpg') or file_path.lower().endswith('.png')):
                file_path += ".jpg"
                
            # حفظ الملف
            cv2.imwrite(file_path, self.processed_image)
            print(f"Image saved to: {file_path}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = CamScannerPro()
    window.show()
    sys.exit(app.exec_())
