import sys
import os
import cv2
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QHBoxLayout, 
                             QVBoxLayout, QPushButton, QLabel, QListWidget, 
                             QListWidgetItem, QCheckBox, QFileDialog, QFrame, 
                             QGridLayout, QGraphicsDropShadowEffect)
from PyQt5.QtGui import QImage, QPixmap, QColor
from PyQt5.QtCore import Qt, QSize

# ==========================================
# 1. خوارزميات المعالجة (CamScanner Magic Engine)
# ==========================================

class CamScannerEngine:
    @staticmethod
    def magic_pro_enhance(img):
        """
        خوارزمية CamScanner Magic Color الحقيقية:
        - إزالة الظلال والتجاعيد مع الحفاظ على ألوان الأختام والشعارات
        - تبييض الورقة لتصبح ناصعة البيض بدون نويز أو خربشة
        """
        if img is None:
            return None

        # التحويل لنظام ألوان LAB لعزل الإضاءة عن الألوان
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)

        # حساب خلفية الورقة والظلال لإزالتها
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (35, 35))
        background = cv2.morphologyEx(l_channel, cv2.MORPH_CLOSE, kernel)

        # قسمة الصورة على الخلفية لتوحيد بياض الورقة
        norm_l = cv2.divide(l_channel, background, scale=255)

        # ضبط التباين لإبراز النصوص وتصفية الورقة
        norm_l = cv2.normalize(norm_l, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        enhanced_l = cv2.convertScaleAbs(norm_l, alpha=1.6, beta=-40)

        # دمج ألوان الأختام والشعارات الأصلية
        enhanced_lab = cv2.merge((enhanced_l, a_channel, b_channel))
        result = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

        # زيادة حدة النصوص (Sharpening)
        gaussian = cv2.GaussianBlur(result, (0, 0), 3)
        sharpened = cv2.addWeighted(result, 1.5, gaussian, -0.5, 0)

        return sharpened

    @staticmethod
    def auto_detect_and_crop(image):
        if image is None:
            return None
            
        orig = image.copy()
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edged = cv2.Canny(blur, 30, 120)

        cnts, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]

        doc_cnt = None
        for c in cnts:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            if len(approx) == 4 and cv2.contourArea(c) > (image.shape[0] * image.shape[1] * 0.15):
                doc_cnt = approx
                break

        if doc_cnt is not None:
            return CamScannerEngine.four_point_transform(orig, doc_cnt.reshape(4, 2))
        return orig

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
        rect = CamScannerEngine.order_points(pts)
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
        return cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    @staticmethod
    def rotate_image(image, angle):
        if angle == 90:
            return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        elif angle == -90:
            return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        elif angle == 180:
            return cv2.rotate(image, cv2.ROTATE_180)
        return image

    @staticmethod
    def apply_enhance(image):
        return cv2.convertScaleAbs(image, alpha=1.2, beta=10)

    @staticmethod
    def apply_lighten(image):
        return cv2.convertScaleAbs(image, alpha=1.1, beta=40)

    @staticmethod
    def apply_bw(image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        return cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)


# ==========================================
# 2. الواجهة الرسومية المطابقة (CamScanner UI)
# ==========================================

class CamScannerUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("CamScanner PC Studio Pro")
        self.resize(1280, 800)
        
        self.images_list = []
        self.current_idx = -1

        self.init_ui()

    def init_ui(self):
        self.setStyleSheet("""
            QMainWindow { background-color: #F0F2F5; }
            QWidget { font-family: 'Segoe UI', Arial, sans-serif; }
        """)

        central_widget = QWidget()
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # 1. Top Bar
        top_bar = QWidget()
        top_bar.setFixedHeight(65)
        top_bar.setStyleSheet("background-color: #F0F2F5;")
        top_layout = QHBoxLayout(top_bar)

        self.btn_import = QPushButton("   Import Images")
        self.btn_import.setFixedHeight(42)
        self.btn_import.setCursor(Qt.PointingHandCursor)
        self.btn_import.setStyleSheet("""
            QPushButton {
                background-color: #FFFFFF;
                border: 1px solid #DCDFE6;
                border-radius: 8px;
                padding: 0 25px;
                font-size: 15px;
                font-weight: 600;
                color: #2C3E50;
            }
            QPushButton:hover { background-color: #F8F9FA; border-color: #B0B5C0; }
        """)
        self.btn_import.clicked.connect(self.import_images)

        top_layout.addStretch()
        top_layout.addWidget(self.btn_import)
        top_layout.addStretch()

        main_layout.addWidget(top_bar)

        # 2. Main Content
        content_widget = QWidget()
        content_layout = QHBoxLayout(content_widget)
        content_layout.setContentsMargins(15, 0, 15, 15)
        content_layout.setSpacing(15)

        # Left Gallery Sidebar
        left_panel = QFrame()
        left_panel.setFixedWidth(140)
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)

        self.thumb_list = QListWidget()
        self.thumb_list.setStyleSheet("""
            QListWidget { background: transparent; border: none; }
            QListWidget::item {
                background-color: #FFFFFF;
                border: 2px solid transparent;
                border-radius: 6px;
                margin-bottom: 12px;
                padding: 5px;
            }
            QListWidget::item:selected {
                border-color: #00A896;
                background-color: #E6F7F5;
            }
        """)
        self.thumb_list.setIconSize(QSize(100, 130))
        self.thumb_list.currentRowChanged.connect(self.select_page)

        left_layout.addWidget(self.thumb_list)
        content_layout.addWidget(left_panel)

        # Center Display Area
        center_panel = QFrame()
        center_panel.setStyleSheet("background-color: #E6E8EC; border-radius: 8px;")
        center_layout = QVBoxLayout(center_panel)
        center_layout.setContentsMargins(20, 20, 20, 20)

        self.image_display = QLabel("اضغط Import Images لتحميل المستندات")
        self.image_display.setAlignment(Qt.AlignCenter)
        self.image_display.setStyleSheet("""
            QLabel {
                background-color: #FFFFFF;
                border: 1px solid #DCDFE6;
                border-radius: 4px;
                color: #909399;
                font-size: 15px;
            }
        """)
        
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(15)
        shadow.setColor(QColor(0, 0, 0, 30))
        shadow.setOffset(0, 4)
        self.image_display.setGraphicsEffect(shadow)

        center_layout.addWidget(self.image_display)
        content_layout.addWidget(center_panel, stretch=1)

        # Right Control Panel
        right_panel = QFrame()
        right_panel.setFixedWidth(260)
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(5, 0, 5, 0)
        right_layout.setSpacing(12)

        # Action Buttons
        tools_grid = QHBoxLayout()
        self.btn_rot_l = QPushButton("⟲")
        self.btn_rot_r = QPushButton("⟳")
        self.btn_flip_h = QPushButton("⇎")
        self.btn_flip_v = QPushButton("⇕")

        for btn in [self.btn_rot_l, self.btn_rot_r, self.btn_flip_h, self.btn_flip_v]:
            btn.setStyleSheet("""
                QPushButton {
                    background-color: #FFFFFF;
                    border: 1px solid #DCDFE6;
                    border-radius: 6px;
                    font-size: 16px;
                    height: 38px;
                    color: #409EFF;
                }
                QPushButton:hover { background-color: #F2F6FC; }
            """)
            btn.setCursor(Qt.PointingHandCursor)
            tools_grid.addWidget(btn)

        self.btn_rot_l.clicked.connect(lambda: self.rotate_current(-90))
        self.btn_rot_r.clicked.connect(lambda: self.rotate_current(90))

        right_layout.addLayout(tools_grid)

        # Crop Button
        self.btn_crop = QPushButton(" ✂   Crop")
        self.btn_crop.setFixedHeight(40)
        self.btn_crop.setCursor(Qt.PointingHandCursor)
        self.btn_crop.setStyleSheet("""
            QPushButton {
                background-color: #FFFFFF;
                border: 1px solid #DCDFE6;
                border-radius: 6px;
                font-size: 14px;
                font-weight: bold;
                color: #303133;
            }
            QPushButton:hover { background-color: #F5F7FA; }
        """)
        self.btn_crop.clicked.connect(self.crop_current)
        right_layout.addWidget(self.btn_crop)

        # Filter Grid
        filter_grid = QGridLayout()
        filter_grid.setSpacing(10)

        self.filter_btns = {}
        filters_data = [
            ("original", "Original", "📄"),
            ("enhance", "Enhance", "🏔️"),
            ("magic", "Magic Pro", "✨ AI"),
            ("lighten", "Lighten", "☀️"),
            ("bw", "B&W", "◐")
        ]

        row, col = 0, 0
        for key, name, icon_str in filters_data:
            btn = QPushButton(f"{icon_str}\n{name}")
            btn.setFixedHeight(75)
            btn.setCursor(Qt.PointingHandCursor)
            
            if key == "magic":
                btn.setStyleSheet("""
                    QPushButton {
                        background-color: #7FE5D9;
                        color: #007A6C;
                        border: 2px solid #00A896;
                        border-radius: 12px;
                        font-size: 13px;
                        font-weight: bold;
                    }
                """)
            else:
                btn.setStyleSheet("""
                    QPushButton {
                        background-color: #E8ECEF;
                        color: #4A5568;
                        border: 1px solid #D2D6DC;
                        border-radius: 12px;
                        font-size: 13px;
                        font-weight: 600;
                    }
                    QPushButton:hover { background-color: #E2E8F0; }
                """)
            
            btn.clicked.connect(lambda checked, k=key: self.set_filter(k))
            filter_grid.addWidget(btn, row, col)
            self.filter_btns[key] = btn

            col += 1
            if col > 1:
                col = 0
                row += 1

        right_layout.addLayout(filter_grid)
        right_layout.addStretch()

        # Save Bar
        self.chk_apply_all = QCheckBox("Apply to All Pages")
        self.chk_apply_all.setChecked(True)
        self.chk_apply_all.setStyleSheet("font-size: 13px; color: #2C3E50; font-weight: 500;")
        right_layout.addWidget(self.chk_apply_all)

        self.btn_save = QPushButton("Confirm")
        self.btn_save.setFixedHeight(45)
        self.btn_save.setCursor(Qt.PointingHandCursor)
        self.btn_save.setStyleSheet("""
            QPushButton {
                background-color: #00B094;
                color: white;
                font-size: 16px;
                font-weight: bold;
                border-radius: 6px;
            }
            QPushButton:hover { background-color: #00967E; }
        """)
        self.btn_save.clicked.connect(self.save_images)
        right_layout.addWidget(self.btn_save)

        content_layout.addWidget(right_panel)
        main_layout.addWidget(content_widget)
        self.setCentralWidget(central_widget)

    def import_images(self):
        file_paths, _ = QFileDialog.getOpenFileNames(
            self, "Select Images", "", "Image Files (*.png *.jpg *.jpeg *.bmp)"
        )
        if not file_paths:
            return

        for path in file_paths:
            img = cv2.imread(path)
            if img is None:
                continue

            cropped = CamScannerEngine.auto_detect_and_crop(img)
            enhanced = CamScannerEngine.magic_pro_enhance(cropped)

            data = {'orig': cropped, 'processed': enhanced, 'filter': 'magic'}
            self.images_list.append(data)
            
            idx = len(self.images_list)
            item = QListWidgetItem(f"{idx}")
            item.setTextAlignment(Qt.AlignCenter)
            self.thumb_list.addItem(item)

        if self.images_list:
            self.thumb_list.setCurrentRow(len(self.images_list) - 1)

    def select_page(self, row):
        if 0 <= row < len(self.images_list):
            self.current_idx = row
            data = self.images_list[row]
            self.display_image(data['processed'])
            self.update_filter_ui(data['filter'])

    def display_image(self, img):
        if img is None:
            return

        h, w, c = img.shape
        rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        q_img = QImage(rgb_image.data, w, h, 3 * w, QImage.Format_RGB888)

        pixmap = QPixmap.fromImage(q_img)
        view_size = self.image_display.size()
        scaled_pixmap = pixmap.scaled(
            view_size.width() - 20, view_size.height() - 20, 
            Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.image_display.setPixmap(scaled_pixmap)

    def set_filter(self, filter_key):
        if self.current_idx < 0:
            return

        target_indices = range(len(self.images_list)) if self.chk_apply_all.isChecked() else [self.current_idx]

        for idx in target_indices:
            data = self.images_list[idx]
            data['filter'] = filter_key
            orig = data['orig']

            if filter_key == "original":
                data['processed'] = orig.copy()
            elif filter_key == "enhance":
                data['processed'] = CamScannerEngine.apply_enhance(orig)
            elif filter_key == "magic":
                data['processed'] = CamScannerEngine.magic_pro_enhance(orig)
            elif filter_key == "lighten":
                data['processed'] = CamScannerEngine.apply_lighten(orig)
            elif filter_key == "bw":
                data['processed'] = CamScannerEngine.apply_bw(orig)

        self.display_image(self.images_list[self.current_idx]['processed'])
        self.update_filter_ui(filter_key)

    def update_filter_ui(self, active_key):
        for key, btn in self.filter_btns.items():
            if key == active_key:
                btn.setStyleSheet("""
                    QPushButton {
                        background-color: #7FE5D9;
                        color: #007A6C;
                        border: 2px solid #00A896;
                        border-radius: 12px;
                        font-size: 13px;
                        font-weight: bold;
                    }
                """)
            else:
                btn.setStyleSheet("""
                    QPushButton {
                        background-color: #E8ECEF;
                        color: #4A5568;
                        border: 1px solid #D2D6DC;
                        border-radius: 12px;
                        font-size: 13px;
                        font-weight: 600;
                    }
                    QPushButton:hover { background-color: #E2E8F0; }
                """)

    def rotate_current(self, angle):
        if self.current_idx < 0:
            return
        data = self.images_list[self.current_idx]
        data['orig'] = CamScannerEngine.rotate_image(data['orig'], angle)
        self.set_filter(data['filter'])

    def crop_current(self):
        if self.current_idx < 0:
            return
        data = self.images_list[self.current_idx]
        data['orig'] = CamScannerEngine.auto_detect_and_crop(data['orig'])
        self.set_filter(data['filter'])

    def save_images(self):
        if not self.images_list:
            return

        if len(self.images_list) == 1:
            path, _ = QFileDialog.getSaveFileName(self, "Save Document", "", "JPEG Image (*.jpg);;PNG Image (*.png)")
            if path:
                cv2.imwrite(path, self.images_list[0]['processed'])
        else:
            dir_path = QFileDialog.getExistingDirectory(self, "Select Save Folder")
            if dir_path:
                for idx, data in enumerate(self.images_list):
                    out_path = os.path.join(dir_path, f"Page_{idx+1}.jpg")
                    cv2.imwrite(out_path, data['processed'])

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = CamScannerUI()
    window.show()
    sys.exit(app.exec_())
