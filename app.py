import sys
import os
import cv2
import numpy as np

from PySide6.QtCore import Qt, QPointF, Signal
from PySide6.QtGui import (
    QImage, QPixmap, QPainter, QPen, QBrush,
    QColor, QFont
)
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton,
    QFileDialog, QVBoxLayout, QHBoxLayout, QSlider,
    QMessageBox, QFrame, QComboBox
)

from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader


# ============================================================
# Image utilities
# ============================================================

def order_points(points):
    pts = np.array(points, dtype=np.float32)

    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).reshape(-1)

    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(d)]
    bl = pts[np.argmax(d)]

    return np.array([tl, tr, br, bl], dtype=np.float32)


def detect_document_corners(image):
    """
    Detect the largest document-like contour.
    Returns 4 corners or None.
    """

    original = image.copy()

    h, w = image.shape[:2]

    scale = 1000.0 / max(h, w)

    if scale < 1:
        small = cv2.resize(
            image,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_AREA
        )
    else:
        small = image.copy()

    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

    # Improve local contrast
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(gray, 50, 150)

    kernel = np.ones((5, 5), np.uint8)
    edges = cv2.morphologyEx(
        edges,
        cv2.MORPH_CLOSE,
        kernel
    )

    contours, _ = cv2.findContours(
        edges,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    contours = sorted(
        contours,
        key=cv2.contourArea,
        reverse=True
    )

    image_area = small.shape[0] * small.shape[1]

    for contour in contours[:30]:

        area = cv2.contourArea(contour)

        if area < image_area * 0.15:
            continue

        perimeter = cv2.arcLength(contour, True)

        approx = cv2.approxPolyDP(
            contour,
            0.02 * perimeter,
            True
        )

        if len(approx) == 4:

            pts = approx.reshape(4, 2).astype(np.float32)

            # Return coordinates to original image
            if scale < 1:
                pts /= scale

            return order_points(pts)

    # Fallback: image boundaries
    h, w = original.shape[:2]

    margin_x = w * 0.03
    margin_y = h * 0.03

    return np.array([
        [margin_x, margin_y],
        [w - margin_x, margin_y],
        [w - margin_x, h - margin_y],
        [margin_x, h - margin_y]
    ], dtype=np.float32)


def perspective_transform(image, corners):

    pts = order_points(corners)

    tl, tr, br, bl = pts

    width_a = np.linalg.norm(br - bl)
    width_b = np.linalg.norm(tr - tl)

    max_width = int(max(width_a, width_b))

    height_a = np.linalg.norm(tr - br)
    height_b = np.linalg.norm(tl - bl)

    max_height = int(max(height_a, height_b))

    max_width = max(300, max_width)
    max_height = max(300, max_height)

    destination = np.array([
        [0, 0],
        [max_width - 1, 0],
        [max_width - 1, max_height - 1],
        [0, max_height - 1]
    ], dtype=np.float32)

    matrix = cv2.getPerspectiveTransform(
        pts,
        destination
    )

    result = cv2.warpPerspective(
        image,
        matrix,
        (max_width, max_height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE
    )

    return result


# ============================================================
# CamScanner-style enhancement
# ============================================================

def enhance_document(image, mode="Document",
                      brightness=0,
                      contrast=100,
                      sharpness=50):

    result = image.copy()

    if mode == "Original":
        return result

    # ----------------------------------------
    # Resize only for processing if extremely
    # huge, but keep final dimensions
    # ----------------------------------------

    # ----------------------------------------
    # Remove uneven illumination
    # ----------------------------------------

    gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)

    background = cv2.GaussianBlur(
        gray,
        (0, 0),
        sigmaX=25,
        sigmaY=25
    )

    background = np.maximum(
        background,
        1
    )

    normalized = cv2.divide(
        gray,
        background,
        scale=255
    )

    normalized = cv2.normalize(
        normalized,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    )

    if mode == "B&W":

        # Adaptive threshold gives better
        # document results than simple threshold

        bw = cv2.adaptiveThreshold(
            normalized,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            31,
            12
        )

        # Remove tiny noise
        kernel = np.ones((2, 2), np.uint8)

        bw = cv2.morphologyEx(
            bw,
            cv2.MORPH_OPEN,
            kernel
        )

        result = cv2.cvtColor(
            bw,
            cv2.COLOR_GRAY2BGR
        )

    elif mode == "Document":

        # Strong but natural document processing

        lab = cv2.cvtColor(
            result,
            cv2.COLOR_BGR2LAB
        )

        l, a, b = cv2.split(lab)

        clahe = cv2.createCLAHE(
            clipLimit=2.0,
            tileGridSize=(8, 8)
        )

        l = clahe.apply(l)

        lab = cv2.merge([l, a, b])

        result = cv2.cvtColor(
            lab,
            cv2.COLOR_LAB2BGR
        )

        # Slight denoise
        result = cv2.fastNlMeansDenoisingColored(
            result,
            None,
            3,
            3,
            7,
            21
        )

    elif mode == "Color":

        lab = cv2.cvtColor(
            result,
            cv2.COLOR_BGR2LAB
        )

        l, a, b = cv2.split(lab)

        clahe = cv2.createCLAHE(
            clipLimit=2.2,
            tileGridSize=(8, 8)
        )

        l = clahe.apply(l)

        lab = cv2.merge([l, a, b])

        result = cv2.cvtColor(
            lab,
            cv2.COLOR_LAB2BGR
        )

    # ----------------------------------------
    # Brightness
    # ----------------------------------------

    if brightness != 0:

        result = cv2.convertScaleAbs(
            result,
            alpha=1.0,
            beta=brightness
        )

    # ----------------------------------------
    # Contrast
    # ----------------------------------------

    alpha = max(0.5, contrast / 100.0)

    result = cv2.convertScaleAbs(
        result,
        alpha=alpha,
        beta=0
    )

    # ----------------------------------------
    # Sharpen
    # ----------------------------------------

    if sharpness > 0:

        amount = sharpness / 100.0

        blurred = cv2.GaussianBlur(
            result,
            (0, 0),
            2
        )

        result = cv2.addWeighted(
            result,
            1.0 + amount,
            blurred,
            -amount,
            0
        )

    return result


# ============================================================
# Image conversion
# ============================================================

def cv_to_qpixmap(image):

    rgb = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    h, w, ch = rgb.shape

    bytes_per_line = ch * w

    qimage = QImage(
        rgb.data,
        w,
        h,
        bytes_per_line,
        QImage.Format_RGB888
    ).copy()

    return QPixmap.fromImage(qimage)


# ============================================================
# Interactive document viewer
# ============================================================

class DocumentViewer(QWidget):

    cornersChanged = Signal()

    def __init__(self):

        super().__init__()

        self.setMinimumSize(700, 500)

        self.image = None
        self.display_image = None

        self.corners = []

        self.dragging = -1

        self.margin = 30

    def set_image(self, image):

        self.image = image.copy()

        self.update_display()

    def set_corners(self, corners):

        self.corners = [
            QPointF(float(x), float(y))
            for x, y in corners
        ]

        self.update()

    def update_display(self):

        if self.image is None:
            return

        self.display_image = cv_to_qpixmap(
            self.image
        )

        self.update()

    def image_to_widget(self, point):

        if self.image is None:
            return QPointF()

        iw = self.image.shape[1]
        ih = self.image.shape[0]

        available_w = self.width() - 2 * self.margin
        available_h = self.height() - 2 * self.margin

        scale = min(
            available_w / iw,
            available_h / ih
        )

        x = self.margin + point.x() * scale
        y = self.margin + point.y() * scale

        return QPointF(x, y)

    def widget_to_image(self, point):

        if self.image is None:
            return QPointF()

        iw = self.image.shape[1]
        ih = self.image.shape[0]

        available_w = self.width() - 2 * self.margin
        available_h = self.height() - 2 * self.margin

        scale = min(
            available_w / iw,
            available_h / ih
        )

        x = (point.x() - self.margin) / scale
        y = (point.y() - self.margin) / scale

        x = max(0, min(iw - 1, x))
        y = max(0, min(ih - 1, y))

        return QPointF(x, y)

    def paintEvent(self, event):

        painter = QPainter(self)

        painter.fillRect(
            self.rect(),
            QColor("#111827")
        )

        if self.display_image is None:
            painter.setPen(QColor("#9CA3AF"))
            painter.setFont(QFont("Arial", 18))
            painter.drawText(
                self.rect(),
                Qt.AlignCenter,
                "Import File لبدء المسح"
            )
            return

        iw = self.display_image.width()
        ih = self.display_image.height()

        available_w = self.width() - 2 * self.margin
        available_h = self.height() - 2 * self.margin

        scale = min(
            available_w / iw,
            available_h / ih
        )

        new_w = int(iw * scale)
        new_h = int(ih * scale)

        x = (self.width() - new_w) // 2
        y = (self.height() - new_h) // 2

        target = self.display_image.scaled(
            new_w,
            new_h,
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )

        painter.drawPixmap(
            x,
            y,
            target
        )

        # ----------------------------------------
        # Draw corners
        # ----------------------------------------

        if len(self.corners) == 4:

            points = [
                self.image_to_widget(p)
                for p in self.corners
            ]

            pen = QPen(
                QColor("#22C55E"),
                3
            )

            painter.setPen(pen)

            for i in range(4):

                p1 = points[i]
                p2 = points[(i + 1) % 4]

                painter.drawLine(
                    p1,
                    p2
                )

            painter.setPen(
                QPen(
                    QColor("#FFFFFF"),
                    2
                )
            )

            painter.setBrush(
                QBrush(
                    QColor("#22C55E")
                )
            )

            for p in points:

                painter.drawEllipse(
                    p,
                    9,
                    9
                )

    def mousePressEvent(self, event):

        if not self.corners:
            return

        pos = event.position()

        for i, corner in enumerate(self.corners):

            wp = self.image_to_widget(
                corner
            )

            distance = (
                (wp.x() - pos.x()) ** 2 +
                (wp.y() - pos.y()) ** 2
            ) ** 0.5

            if distance < 25:

                self.dragging = i

                return

    def mouseMoveEvent(self, event):

        if self.dragging < 0:
            return

        new_point = self.widget_to_image(
            event.position()
        )

        self.corners[
            self.dragging
        ] = new_point

        self.update()

        self.cornersChanged.emit()

    def mouseReleaseEvent(self, event):

        self.dragging = -1


# ============================================================
# Main Window
# ============================================================

class ScannerApp(QMainWindow):

    def __init__(self):

        super().__init__()

        self.setWindowTitle(
            "ScanPro - Document Scanner"
        )

        self.resize(1400, 850)

        self.original_image = None
        self.scanned_image = None
        self.corners = None

        self.build_ui()

    # --------------------------------------------------------

    def build_ui(self):

        central = QWidget()

        self.setCentralWidget(
            central
        )

        main = QHBoxLayout(
            central
        )

        main.setContentsMargins(
            0, 0, 0, 0
        )

        # ================================================
        # LEFT TOOLBAR
        # ================================================

        sidebar = QFrame()

        sidebar.setFixedWidth(230)

        sidebar.setStyleSheet("""
            QFrame {
                background: #111827;
            }

            QPushButton {
                background: #1F2937;
                color: white;
                border: none;
                border-radius: 10px;
                padding: 13px;
                font-size: 14px;
                text-align: left;
            }

            QPushButton:hover {
                background: #374151;
            }

            QComboBox {
                background: #1F2937;
                color: white;
                border: none;
                border-radius: 10px;
                padding: 10px;
            }

            QLabel {
                color: #D1D5DB;
            }

            QSlider {
                padding: 5px;
            }
        """)

        side = QVBoxLayout(
            sidebar
        )

        title = QLabel(
            "SCAN PRO"
        )

        title.setFont(
            QFont("Arial", 22, QFont.Bold)
        )

        title.setStyleSheet(
            "color:white;"
        )

        side.addWidget(title)

        subtitle = QLabel(
            "Document Scanner"
        )

        side.addWidget(subtitle)

        side.addSpacing(20)

        import_btn = QPushButton(
            "📂   Import File"
        )

        import_btn.clicked.connect(
            self.import_file
        )

        side.addWidget(
            import_btn
        )

        save_btn = QPushButton(
            "💾   Save Image"
        )

        save_btn.clicked.connect(
            self.save_image
        )

        side.addWidget(
            save_btn
        )

        pdf_btn = QPushButton(
            "📄   Export PDF"
        )

        pdf_btn.clicked.connect(
            self.export_pdf
        )

        side.addWidget(
            pdf_btn
        )

        side.addSpacing(20)

        mode_label = QLabel(
            "Enhancement"
        )

        side.addWidget(
            mode_label
        )

        self.mode = QComboBox()

        self.mode.addItems([
            "Document",
            "B&W",
            "Color",
            "Original"
        ])

        self.mode.currentTextChanged.connect(
            self.process_image
        )

        side.addWidget(
            self.mode
        )

        side.addSpacing(15)

        # Brightness

        side.addWidget(
            QLabel("Brightness")
        )

        self.brightness = QSlider(
            Qt.Horizontal
        )

        self.brightness.setMinimum(-50)
        self.brightness.setMaximum(50)
        self.brightness.setValue(0)

        self.brightness.valueChanged.connect(
            self.process_image
        )

        side.addWidget(
            self.brightness
        )

        # Contrast

        side.addWidget(
            QLabel("Contrast")
        )

        self.contrast = QSlider(
            Qt.Horizontal
        )

        self.contrast.setMinimum(50)
        self.contrast.setMaximum(160)
        self.contrast.setValue(100)

        self.contrast.valueChanged.connect(
            self.process_image
        )

        side.addWidget(
            self.contrast
        )

        # Sharpness

        side.addWidget(
            QLabel("Sharpness")
        )

        self.sharpness = QSlider(
            Qt.Horizontal
        )

        self.sharpness.setMinimum(0)
        self.sharpness.setMaximum(100)
        self.sharpness.setValue(50)

        self.sharpness.valueChanged.connect(
            self.process_image
        )

        side.addWidget(
            self.sharpness
        )

        side.addStretch()

        reset_btn = QPushButton(
            "↺   Reset"
        )

        reset_btn.clicked.connect(
            self.reset_image
        )

        side.addWidget(
            reset_btn
        )

        main.addWidget(
            sidebar
        )

        # ================================================
        # RIGHT AREA
        # ================================================

        right = QWidget()

        right_layout = QVBoxLayout(
            right
        )

        right_layout.setContentsMargins(
            0, 0, 0, 0
        )

        # Header

        header = QFrame()

        header.setFixedHeight(
            65
        )

        header.setStyleSheet("""
            QFrame {
                background: #FFFFFF;
                border-bottom: 1px solid #E5E7EB;
            }

            QLabel {
                color: #111827;
                font-size: 15px;
            }
        """)

        header_layout = QHBoxLayout(
            header
        )

        self.status = QLabel(
            "Import a document to begin"
        )

        header_layout.addWidget(
            self.status
        )

        header_layout.addStretch()

        header_layout.addWidget(
            QLabel("Auto Scan ✓")
        )

        right_layout.addWidget(
            header
        )

        # Viewer

        self.viewer = DocumentViewer()

        self.viewer.cornersChanged.connect(
            self.process_image
        )

        right_layout.addWidget(
            self.viewer,
            1
        )

        # Bottom

        bottom = QFrame()

        bottom.setFixedHeight(
            65
        )

        bottom.setStyleSheet("""
            QFrame {
                background: white;
                border-top: 1px solid #E5E7EB;
            }

            QPushButton {
                background: #111827;
                color: white;
                border-radius: 10px;
                padding: 10px 20px;
            }

            QPushButton:hover {
                background: #374151;
            }
        """)

        bottom_layout = QHBoxLayout(
            bottom
        )

        bottom_layout.addStretch()

        save_bottom = QPushButton(
            "Save Scan"
        )

        save_bottom.clicked.connect(
            self.save_image
        )

        bottom_layout.addWidget(
            save_bottom
        )

        right_layout.addWidget(
            bottom
        )

        main.addWidget(
            right,
            1
        )

    # ====================================================
    # Import
    # ====================================================

    def import_file(self):

        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Import Document",
            "",
            "Images (*.jpg *.jpeg *.png *.bmp *.tif *.tiff)"
        )

        if not filename:
            return

        image = cv2.imread(
            filename,
            cv2.IMREAD_COLOR
        )

        if image is None:

            QMessageBox.critical(
                self,
                "Error",
                "Unable to open image."
            )

            return

        self.original_image = image

        self.corners = detect_document_corners(
            image
        )

        self.viewer.set_image(
            image
        )

        self.viewer.set_corners(
            self.corners
        )

        self.status.setText(
            "Document detected — processing automatically"
        )

        self.process_image()

    # ====================================================
    # Live processing
    # ====================================================

    def process_image(self):

        if self.original_image is None:
            return

        corners = self.corners

        if self.viewer.corners:

            corners = np.array([
                [p.x(), p.y()]
                for p in self.viewer.corners
            ], dtype=np.float32)

        if corners is None:
            return

        # Perspective correction

        scanned = perspective_transform(
            self.original_image,
            corners
        )

        # Enhancement

        scanned = enhance_document(
            scanned,
            self.mode.currentText(),
            self.brightness.value(),
            self.contrast.value(),
            self.sharpness.value()
        )

        self.scanned_image = scanned

        # Display the processed result immediately
        self.viewer.set_image(
            scanned
        )

        # Keep corners corresponding to
        # current source image only when
        # user is dragging.
        #
        # After processing, viewer shows
        # processed image, therefore remove
        # corner overlay temporarily.

        self.viewer.corners = []

        self.status.setText(
            f"Processed: {scanned.shape[1]} × {scanned.shape[0]} px"
        )

    # ====================================================
    # Reset
    # ====================================================

    def reset_image(self):

        if self.original_image is None:
            return

        self.corners = detect_document_corners(
            self.original_image
        )

        self.viewer.set_image(
            self.original_image
        )

        self.viewer.set_corners(
            self.corners
        )

        self.mode.setCurrentText(
            "Document"
        )

        self.brightness.setValue(0)
        self.contrast.setValue(100)
        self.sharpness.setValue(50)

        self.process_image()

    # ====================================================
    # Save
    # ====================================================

    def save_image(self):

        if self.scanned_image is None:

            QMessageBox.warning(
                self,
                "No document",
                "Import a document first."
            )

            return

        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Save Scan",
            "",
            "JPEG Image (*.jpg);;PNG Image (*.png)"
        )

        if not filename:
            return

        ext = os.path.splitext(
            filename
        )[1].lower()

        if ext == ".png":

            cv2.imwrite(
                filename,
                self.scanned_image,
                [
                    cv2.IMWRITE_PNG_COMPRESSION,
                    1
                ]
            )

        else:

            if not filename.lower().endswith(
                ".jpg"
            ):
                filename += ".jpg"

            cv2.imwrite(
                filename,
                self.scanned_image,
                [
                    cv2.IMWRITE_JPEG_QUALITY,
                    98
                ]
            )

        QMessageBox.information(
            self,
            "Saved",
            "Document saved successfully."
        )

    # ====================================================
    # PDF
    # ====================================================

    def export_pdf(self):

        if self.scanned_image is None:

            QMessageBox.warning(
                self,
                "No document",
                "Import a document first."
            )

            return

        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Export PDF",
            "",
            "PDF (*.pdf)"
        )

        if not filename:
            return

        if not filename.lower().endswith(
            ".pdf"
        ):
            filename += ".pdf"

        # Temporary high-quality JPEG
        temp = os.path.join(
            os.path.dirname(filename),
            "_scan_temp.jpg"
        )

        cv2.imwrite(
            temp,
            self.scanned_image,
            [
                cv2.IMWRITE_JPEG_QUALITY,
                98
            ]
        )

        h, w = self.scanned_image.shape[:2]

        pdf = canvas.Canvas(
            filename,
            pagesize=(w, h)
        )

        pdf.drawImage(
            ImageReader(temp),
            0,
            0,
            width=w,
            height=h
        )

        pdf.showPage()
        pdf.save()

        try:
            os.remove(temp)
        except:
            pass

        QMessageBox.information(
            self,
            "PDF",
            "PDF exported successfully."
        )


# ============================================================
# Run
# ============================================================

if __name__ == "__main__":

    app = QApplication(sys.argv)

    app.setStyle(
        "Fusion"
    )

    window = ScannerApp()

    window.show()

    sys.exit(
        app.exec()
    )
