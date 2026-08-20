import sys
import os
import cv2
import numpy as np

from PySide6.QtCore import Qt, QPointF, Signal, QTimer
from PySide6.QtGui import (
    QImage, QPixmap, QPainter, QPen, QBrush,
    QColor, QFont
)
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel,
    QPushButton, QFileDialog, QVBoxLayout, QHBoxLayout,
    QSlider, QMessageBox, QComboBox, QFrame,
    QSizePolicy
)

from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader


# ============================================================
# POINT ORDERING
# ============================================================

def order_points(points):
    pts = np.array(points, dtype=np.float32)

    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).reshape(-1)

    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(d)]
    bl = pts[np.argmax(d)]

    return np.array(
        [tl, tr, br, bl],
        dtype=np.float32
    )


# ============================================================
# DOCUMENT DETECTION
# ============================================================

def detect_document_corners(image):

    h, w = image.shape[:2]

    # Work on smaller copy for detection only
    max_side = 1400.0
    scale = min(1.0, max_side / max(h, w))

    if scale < 1.0:
        small = cv2.resize(
            image,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_AREA
        )
    else:
        small = image.copy()

    gray = cv2.cvtColor(
        small,
        cv2.COLOR_BGR2GRAY
    )

    gray = cv2.GaussianBlur(
        gray,
        (5, 5),
        0
    )

    candidates = []

    # --------------------------------------------------------
    # Method 1: white-page threshold
    # --------------------------------------------------------

    _, thresh = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (9, 9)
    )

    thresh = cv2.morphologyEx(
        thresh,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=2
    )

    contours, _ = cv2.findContours(
        thresh,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    candidates.extend(contours)

    # --------------------------------------------------------
    # Method 2: edge detection
    # --------------------------------------------------------

    edges = cv2.Canny(
        gray,
        30,
        120
    )

    edges = cv2.dilate(
        edges,
        np.ones((3, 3), np.uint8),
        iterations=1
    )

    edges = cv2.morphologyEx(
        edges,
        cv2.MORPH_CLOSE,
        np.ones((7, 7), np.uint8),
        iterations=2
    )

    contours2, _ = cv2.findContours(
        edges,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    candidates.extend(contours2)

    image_area = small.shape[0] * small.shape[1]

    best = None
    best_score = -1

    for contour in candidates:

        area = cv2.contourArea(contour)

        if area < image_area * 0.12:
            continue

        perimeter = cv2.arcLength(
            contour,
            True
        )

        if perimeter <= 0:
            continue

        approx = cv2.approxPolyDP(
            contour,
            0.025 * perimeter,
            True
        )

        if len(approx) != 4:
            continue

        pts = approx.reshape(
            4,
            2
        ).astype(np.float32)

        # Convexity
        if not cv2.isContourConvex(
            pts.astype(np.int32)
        ):
            continue

        rect_area = cv2.contourArea(pts)

        if rect_area <= 0:
            continue

        rectangularity = area / rect_area

        if rectangularity < 0.75:
            continue

        # Prefer large objects
        area_score = area / image_area

        # Prefer corners not too close to the image border
        border_penalty = 0

        margin = min(
            small.shape[0],
            small.shape[1]
        ) * 0.015

        for x, y in pts:

            if (
                x < margin or
                y < margin or
                x > small.shape[1] - margin or
                y > small.shape[0] - margin
            ):
                border_penalty += 0.03

        score = (
            area_score * 2.0
            + rectangularity
            - border_penalty
        )

        if score > best_score:

            best_score = score
            best = pts

    if best is not None:

        if scale < 1.0:
            best /= scale

        return order_points(best)

    # --------------------------------------------------------
    # Fallback
    # --------------------------------------------------------

    margin_x = w * 0.04
    margin_y = h * 0.04

    return np.array([
        [margin_x, margin_y],
        [w - margin_x, margin_y],
        [w - margin_x, h - margin_y],
        [margin_x, h - margin_y]
    ], dtype=np.float32)


# ============================================================
# PERSPECTIVE CORRECTION
# ============================================================

def perspective_transform(image, corners):

    pts = order_points(corners)

    tl, tr, br, bl = pts

    width_top = np.linalg.norm(
        tr - tl
    )

    width_bottom = np.linalg.norm(
        br - bl
    )

    height_left = np.linalg.norm(
        bl - tl
    )

    height_right = np.linalg.norm(
        br - tr
    )

    max_width = int(
        max(
            width_top,
            width_bottom
        )
    )

    max_height = int(
        max(
            height_left,
            height_right
        )
    )

    max_width = max(
        300,
        max_width
    )

    max_height = max(
        300,
        max_height
    )

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
        (
            max_width,
            max_height
        ),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE
    )

    return result


# ============================================================
# IMAGE ENHANCEMENT
# ============================================================

def correct_illumination(gray):

    # Estimate background illumination
    background = cv2.GaussianBlur(
        gray,
        (0, 0),
        25
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

    return normalized


def document_enhancement(image):

    # ----------------------------------------
    # LAB contrast improvement
    # ----------------------------------------

    lab = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2LAB
    )

    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    l = clahe.apply(l)

    lab = cv2.merge([
        l,
        a,
        b
    ])

    result = cv2.cvtColor(
        lab,
        cv2.COLOR_LAB2BGR
    )

    # ----------------------------------------
    # Light denoise
    # ----------------------------------------

    result = cv2.fastNlMeansDenoisingColored(
        result,
        None,
        2,
        2,
        7,
        21
    )

    # ----------------------------------------
    # Correct uneven lighting
    # ----------------------------------------

    gray = cv2.cvtColor(
        result,
        cv2.COLOR_BGR2GRAY
    )

    corrected = correct_illumination(
        gray
    )

    # Blend corrected luminance with color
    hsv = cv2.cvtColor(
        result,
        cv2.COLOR_BGR2HSV
    )

    hsv[:, :, 2] = corrected

    result = cv2.cvtColor(
        hsv,
        cv2.COLOR_HSV2BGR
    )

    # ----------------------------------------
    # Mild sharpening
    # ----------------------------------------

    blurred = cv2.GaussianBlur(
        result,
        (0, 0),
        1.2
    )

    result = cv2.addWeighted(
        result,
        1.20,
        blurred,
        -0.20,
        0
    )

    return result


def bw_enhancement(image):

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    # Correct shadows / illumination
    gray = correct_illumination(
        gray
    )

    # Increase text contrast
    clahe = cv2.createCLAHE(
        clipLimit=2.5,
        tileGridSize=(8, 8)
    )

    gray = clahe.apply(
        gray
    )

    # Adaptive threshold
    result = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        11
    )

    # Remove tiny noise
    result = cv2.morphologyEx(
        result,
        cv2.MORPH_OPEN,
        np.ones((2, 2), np.uint8)
    )

    return cv2.cvtColor(
        result,
        cv2.COLOR_GRAY2BGR
    )


def color_enhancement(image):

    lab = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2LAB
    )

    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    l = clahe.apply(l)

    lab = cv2.merge([
        l,
        a,
        b
    ])

    result = cv2.cvtColor(
        lab,
        cv2.COLOR_LAB2BGR
    )

    return result


def enhance_image(
    image,
    mode,
    brightness,
    contrast,
    sharpness
):

    if mode == "Original":

        result = image.copy()

    elif mode == "B&W":

        result = bw_enhancement(
            image
        )

    elif mode == "Color":

        result = color_enhancement(
            image
        )

    else:

        result = document_enhancement(
            image
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

    alpha = contrast / 100.0

    result = cv2.convertScaleAbs(
        result,
        alpha=alpha,
        beta=0
    )

    # ----------------------------------------
    # Sharpness
    # ----------------------------------------

    if sharpness > 0:

        amount = sharpness / 100.0

        blurred = cv2.GaussianBlur(
            result,
            (0, 0),
            1.3
        )

        result = cv2.addWeighted(
            result,
            1.0 + amount * 0.7,
            blurred,
            -amount * 0.7,
            0
        )

    return result


# ============================================================
# CONVERT OPENCV IMAGE TO QPIXMAP
# ============================================================

def cv_to_pixmap(image):

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

    return QPixmap.fromImage(
        qimage
    )


# ============================================================
# ORIGINAL IMAGE VIEWER
# ============================================================

class OriginalViewer(QWidget):

    cornersChanged = Signal()

    def __init__(self):

        super().__init__()

        self.image = None
        self.pixmap = None

        self.corners = []

        self.dragging = -1

        self.setMinimumSize(
            500,
            500
        )

    def set_image(self, image):

        self.image = image

        self.pixmap = cv_to_pixmap(
            image
        )

        self.update()

    def set_corners(self, corners):

        self.corners = [
            QPointF(
                float(p[0]),
                float(p[1])
            )
            for p in corners
        ]

        self.update()

    def image_to_widget(
        self,
        point
    ):

        if self.image is None:
            return QPointF()

        h, w = self.image.shape[:2]

        area_w = self.width() - 30
        area_h = self.height() - 30

        scale = min(
            area_w / w,
            area_h / h
        )

        display_w = w * scale
        display_h = h * scale

        offset_x = (
            self.width() - display_w
        ) / 2

        offset_y = (
            self.height() - display_h
        ) / 2

        return QPointF(
            offset_x + point.x() * scale,
            offset_y + point.y() * scale
        )

    def widget_to_image(
        self,
        point
    ):

        if self.image is None:
            return QPointF()

        h, w = self.image.shape[:2]

        area_w = self.width() - 30
        area_h = self.height() - 30

        scale = min(
            area_w / w,
            area_h / h
        )

        display_w = w * scale
        display_h = h * scale

        offset_x = (
            self.width() - display_w
        ) / 2

        offset_y = (
            self.height() - display_h
        ) / 2

        x = (
            point.x() - offset_x
        ) / scale

        y = (
            point.y() - offset_y
        ) / scale

        x = max(
            0,
            min(w - 1, x)
        )

        y = max(
            0,
            min(h - 1, y)
        )

        return QPointF(
            x,
            y
        )

    def paintEvent(self, event):

        painter = QPainter(
            self
        )

        painter.setRenderHint(
            QPainter.SmoothPixmapTransform
        )

        painter.fillRect(
            self.rect(),
            QColor("#0B1020")
        )

        if self.pixmap is None:

            painter.setPen(
                QColor("#9CA3AF")
            )

            painter.setFont(
                QFont(
                    "Arial",
                    17
                )
            )

            painter.drawText(
                self.rect(),
                Qt.AlignCenter,
                "Import File"
            )

            return

        h, w = self.image.shape[:2]

        area_w = self.width() - 30
        area_h = self.height() - 30

        scale = min(
            area_w / w,
            area_h / h
        )

        display_w = int(
            w * scale
        )

        display_h = int(
            h * scale
        )

        x = (
            self.width()
            - display_w
        ) // 2

        y = (
            self.height()
            - display_h
        ) // 2

        scaled = self.pixmap.scaled(
            display_w,
            display_h,
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )

        painter.drawPixmap(
            x,
            y,
            scaled
        )

        # ----------------------------------------
        # Crop polygon
        # ----------------------------------------

        if len(self.corners) == 4:

            points = [
                self.image_to_widget(p)
                for p in self.corners
            ]

            painter.setPen(
                QPen(
                    QColor("#00D4A8"),
                    3
                )
            )

            for i in range(4):

                painter.drawLine(
                    points[i],
                    points[
                        (i + 1) % 4
                    ]
                )

            # Semi transparent outside
            # crop region

            painter.setBrush(
                QBrush(
                    QColor(
                        0,
                        212,
                        168,
                        25
                    )
                )
            )

            painter.setPen(
                Qt.NoPen
            )

            polygon = [
                points[0],
                points[1],
                points[2],
                points[3]
            ]

            painter.drawPolygon(
                polygon
            )

            # Corner circles

            painter.setBrush(
                QBrush(
                    QColor("#00D4A8")
                )
            )

            painter.setPen(
                QPen(
                    QColor("#FFFFFF"),
                    2
                )
            )

            for p in points:

                painter.drawEllipse(
                    p,
                    9,
                    9
                )

    def mousePressEvent(
        self,
        event
    ):

        if len(self.corners) != 4:
            return

        pos = event.position()

        for i, corner in enumerate(
            self.corners
        ):

            wp = self.image_to_widget(
                corner
            )

            distance = (
                (
                    wp.x()
                    - pos.x()
                ) ** 2
                +
                (
                    wp.y()
                    - pos.y()
                ) ** 2
            ) ** 0.5

            if distance <= 28:

                self.dragging = i

                return

    def mouseMoveEvent(
        self,
        event
    ):

        if self.dragging < 0:
            return

        p = self.widget_to_image(
            event.position()
        )

        self.corners[
            self.dragging
        ] = p

        self.update()

        self.cornersChanged.emit()

    def mouseReleaseEvent(
        self,
        event
    ):

        self.dragging = -1


# ============================================================
# PROCESSED VIEWER
# ============================================================

class ProcessedViewer(QLabel):

    def __init__(self):

        super().__init__()

        self.setAlignment(
            Qt.AlignCenter
        )

        self.setStyleSheet("""
            QLabel {
                background: #080D18;
                border-radius: 10px;
                color: #94A3B8;
            }
        """)

        self.setText(
            "Processed Result"
        )

        self.current_pixmap = None

    def set_image(self, image):

        self.current_pixmap = cv_to_pixmap(
            image
        )

        self.update_display()

    def resizeEvent(
        self,
        event
    ):

        super().resizeEvent(
            event
        )

        self.update_display()

    def update_display(self):

        if self.current_pixmap is None:
            return

        pixmap = self.current_pixmap.scaled(
            self.size() - self.contentsMargins(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )

        self.setPixmap(
            pixmap
        )


# ============================================================
# MAIN APPLICATION
# ============================================================

class ScannerApp(QMainWindow):

    def __init__(self):

        super().__init__()

        self.setWindowTitle(
            "SCAN PRO - Document Scanner"
        )

        self.resize(
            1500,
            900
        )

        self.original_image = None
        self.scanned_image = None
        self.corners = None

        # Used for delayed live processing
        self.process_timer = QTimer()

        self.process_timer.setSingleShot(
            True
        )

        self.process_timer.timeout.connect(
            self.process_image
        )

        self.build_ui()

    # ========================================================
    # UI
    # ========================================================

    def build_ui(self):

        central = QWidget()

        self.setCentralWidget(
            central
        )

        main = QVBoxLayout(
            central
        )

        main.setContentsMargins(
            0,
            0,
            0,
            0
        )

        main.setSpacing(
            0
        )

        # ====================================================
        # TOP BAR
        # ====================================================

        top = QFrame()

        top.setFixedHeight(
            70
        )

        top.setStyleSheet("""
            QFrame {
                background: #0B1325;
            }

            QLabel {
                color: white;
            }

            QPushButton {
                background: #17233A;
                color: white;
                border: none;
                border-radius: 8px;
                padding: 9px 15px;
            }

            QPushButton:hover {
                background: #243450;
            }
        """)

        top_layout = QHBoxLayout(
            top
        )

        logo = QLabel(
            "SCAN PRO"
        )

        logo.setFont(
            QFont(
                "Arial",
                24,
                QFont.Bold
            )
        )

        top_layout.addWidget(
            logo
        )

        subtitle = QLabel(
            "  Document Scanner"
        )

        subtitle.setStyleSheet(
            "color:#94A3B8;"
        )

        top_layout.addWidget(
            subtitle
        )

        top_layout.addSpacing(
            25
        )

        rotate_left = QPushButton(
            "↶  Rotate Left"
        )

        rotate_left.clicked.connect(
            lambda:
            self.rotate_image(
                -1
            )
        )

        top_layout.addWidget(
            rotate_left
        )

        rotate_right = QPushButton(
            "↷  Rotate Right"
        )

        rotate_right.clicked.connect(
            lambda:
            self.rotate_image(
                1
            )
        )

        top_layout.addWidget(
            rotate_right
        )

        top_layout.addStretch()

        original_btn = QPushButton(
            "Original"
        )

        original_btn.clicked.connect(
            self.show_original
        )

        top_layout.addWidget(
            original_btn
        )

        top_layout.addWidget(
            QLabel("  ")
        )

        top_layout.addWidget(
            QLabel(
                "Auto Scan ✓"
            )
        )

        main.addWidget(
            top
        )

        # ====================================================
        # MAIN CONTENT
        # ====================================================

        content = QHBoxLayout()

        content.setContentsMargins(
            10,
            10,
            10,
            10
        )

        content.setSpacing(
            10
        )

        # ====================================================
        # LEFT SIDEBAR
        # ====================================================

        sidebar = QFrame()

        sidebar.setFixedWidth(
            220
        )

        sidebar.setStyleSheet("""
            QFrame {
                background: #0F172A;
                border-radius: 12px;
            }

            QPushButton {
                background: #182338;
                color: #E5E7EB;
                border: none;
                border-radius: 9px;
                padding: 13px;
                text-align: left;
                font-size: 14px;
            }

            QPushButton:hover {
                background: #263650;
            }

            QPushButton#active {
                background: #00A884;
                color: white;
            }

            QLabel {
                color: #CBD5E1;
            }

            QComboBox {
                background: #182338;
                color: white;
                border: none;
                border-radius: 8px;
                padding: 10px;
            }
        """)

        side = QVBoxLayout(
            sidebar
        )

        side.setContentsMargins(
            12,
            15,
            12,
            15
        )

        import_btn = QPushButton(
            "📄   Import File"
        )

        import_btn.setObjectName(
            "active"
        )

        import_btn.clicked.connect(
            self.import_file
        )

        side.addWidget(
            import_btn
        )

        save_btn = QPushButton(
            "▣   Save Image"
        )

        save_btn.clicked.connect(
            self.save_image
        )

        side.addWidget(
            save_btn
        )

        pdf_btn = QPushButton(
            "▤   Export PDF"
        )

        pdf_btn.clicked.connect(
            self.export_pdf
        )

        side.addWidget(
            pdf_btn
        )

        side.addSpacing(
            20
        )

        title = QLabel(
            "Enhancement"
        )

        title.setFont(
            QFont(
                "Arial",
                13,
                QFont.Bold
            )
        )

        side.addWidget(
            title
        )

        self.mode = QComboBox()

        self.mode.addItems([
            "Document",
            "B&W",
            "Color",
            "Original"
        ])

        self.mode.currentTextChanged.connect(
            self.schedule_process
        )

        side.addWidget(
            self.mode
        )

        side.addSpacing(
            12
        )

        # Brightness

        side.addWidget(
            QLabel("Brightness")
        )

        self.brightness = QSlider(
            Qt.Horizontal
        )

        self.brightness.setRange(
            -30,
            30
        )

        self.brightness.setValue(
            8
        )

        self.brightness.valueChanged.connect(
            self.schedule_process
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

        self.contrast.setRange(
            70,
            140
        )

        self.contrast.setValue(
            108
        )

        self.contrast.valueChanged.connect(
            self.schedule_process
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

        self.sharpness.setRange(
            0,
            100
        )

        self.sharpness.setValue(
            22
        )

        self.sharpness.valueChanged.connect(
            self.schedule_process
        )

        side.addWidget(
            self.sharpness
        )

        side.addStretch()

        reset = QPushButton(
            "↻   Reset"
        )

        reset.clicked.connect(
            self.reset
        )

        side.addWidget(
            reset
        )

        content.addWidget(
            sidebar
        )

        # ====================================================
        # ORIGINAL PANEL
        # ====================================================

        original_panel = QFrame()

        original_panel.setStyleSheet("""
            QFrame {
                background: #0B1020;
                border-radius: 12px;
            }
        """)

        original_layout = QVBoxLayout(
            original_panel
        )

        original_title = QLabel(
            "Original / Crop"
        )

        original_title.setStyleSheet(
            "color:white;font-weight:bold;"
        )

        original_layout.addWidget(
            original_title
        )

        self.original_viewer = OriginalViewer()

        self.original_viewer.cornersChanged.connect(
            self.schedule_process
        )

        original_layout.addWidget(
            self.original_viewer
        )

        content.addWidget(
            original_panel,
            3
        )

        # ====================================================
        # PROCESSED PANEL
        # ====================================================

        processed_panel = QFrame()

        processed_panel.setStyleSheet("""
            QFrame {
                background: #0B1020;
                border-radius: 12px;
            }
        """)

        processed_layout = QVBoxLayout(
            processed_panel
        )

        processed_title = QLabel(
            "Processed Result"
        )

        processed_title.setStyleSheet(
            "color:white;font-weight:bold;"
        )

        processed_layout.addWidget(
            processed_title
        )

        self.processed_viewer = ProcessedViewer()

        processed_layout.addWidget(
            self.processed_viewer,
            1
        )

        content.addWidget(
            processed_panel,
            2
        )

        main.addLayout(
            content,
            1
        )

        # ====================================================
        # BOTTOM BAR
        # ====================================================

        bottom = QFrame()

        bottom.setFixedHeight(
            70
        )

        bottom.setStyleSheet("""
            QFrame {
                background: #0B1325;
            }

            QPushButton {
                background: #00A884;
                color: white;
                border: none;
                border-radius: 9px;
                padding: 12px 30px;
                font-weight: bold;
            }

            QPushButton:hover {
                background: #00C49A;
            }

            QLabel {
                color: #CBD5E1;
            }
        """)

        bottom_layout = QHBoxLayout(
            bottom
        )

        self.info = QLabel(
            "Import a document to begin"
        )

        bottom_layout.addWidget(
            self.info
        )

        bottom_layout.addStretch()

        save_scan = QPushButton(
            "▣   Save Scan"
        )

        save_scan.clicked.connect(
            self.save_image
        )

        bottom_layout.addWidget(
            save_scan
        )

        main.addWidget(
            bottom
        )

    # ========================================================
    # IMPORT
    # ========================================================

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
                "Could not open this image."
            )

            return

        self.original_image = image

        self.corners = detect_document_corners(
            image
        )

        self.original_viewer.set_image(
            image
        )

        self.original_viewer.set_corners(
            self.corners
        )

        self.info.setText(
            "Document detected — processing automatically"
        )

        self.process_image()

    # ========================================================
    # LIVE PROCESSING
    # ========================================================

    def schedule_process(self):

        if self.original_image is None:
            return

        # Process shortly after movement
        # instead of processing hundreds
        # of times per second.

        self.process_timer.start(
            80
        )

    def process_image(self):

        if self.original_image is None:
            return

        if len(
            self.original_viewer.corners
        ) != 4:

            return

        corners = np.array([
            [
                p.x(),
                p.y()
            ]
            for p in
            self.original_viewer.corners
        ], dtype=np.float32)

        # ----------------------------------------
        # Perspective crop
        # ----------------------------------------

        cropped = perspective_transform(
            self.original_image,
            corners
        )

        # ----------------------------------------
        # Enhancement
        # ----------------------------------------

        result = enhance_image(
            cropped,
            self.mode.currentText(),
            self.brightness.value(),
            self.contrast.value(),
            self.sharpness.value()
        )

        self.scanned_image = result

        self.processed_viewer.set_image(
            result
        )

        h, w = result.shape[:2]

        self.info.setText(
            f"Scanned document: {w} × {h} px"
        )

    # ========================================================
    # ROTATE
    # ========================================================

    def rotate_image(
        self,
        direction
    ):

        if self.original_image is None:
            return

        if direction > 0:

            self.original_image = cv2.rotate(
                self.original_image,
                cv2.ROTATE_90_CLOCKWISE
            )

        else:

            self.original_image = cv2.rotate(
                self.original_image,
                cv2.ROTATE_90_COUNTERCLOCKWISE
            )

        self.corners = detect_document_corners(
            self.original_image
        )

        self.original_viewer.set_image(
            self.original_image
        )

        self.original_viewer.set_corners(
            self.corners
        )

        self.process_image()

    # ========================================================
    # SHOW ORIGINAL
    # ========================================================

    def show_original(self):

        if self.original_image is None:
            return

        self.original_viewer.set_image(
            self.original_image
        )

        self.original_viewer.set_corners(
            self.corners
        )

    # ========================================================
    # RESET
    # ========================================================

    def reset(self):

        if self.original_image is None:
            return

        self.corners = detect_document_corners(
            self.original_image
        )

        self.original_viewer.set_image(
            self.original_image
        )

        self.original_viewer.set_corners(
            self.corners
        )

        self.mode.setCurrentText(
            "Document"
        )

        self.brightness.setValue(
            8
        )

        self.contrast.setValue(
            108
        )

        self.sharpness.setValue(
            22
        )

        self.process_image()

    # ========================================================
    # SAVE IMAGE
    # ========================================================

    def save_image(self):

        if self.scanned_image is None:

            QMessageBox.warning(
                self,
                "No Scan",
                "Import a document first."
            )

            return

        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Save Scan",
            "ScanPro.jpg",
            "JPEG Image (*.jpg *.jpeg);;PNG Image (*.png)"
        )

        if not filename:
            return

        if filename.lower().endswith(
            ".png"
        ):

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
                (".jpg", ".jpeg")
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
            "Scan saved successfully."
        )

    # ========================================================
    # PDF
    # ========================================================

    def export_pdf(self):

        if self.scanned_image is None:

            QMessageBox.warning(
                self,
                "No Scan",
                "Import a document first."
            )

            return

        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Export PDF",
            "ScanPro.pdf",
            "PDF (*.pdf)"
        )

        if not filename:
            return

        if not filename.lower().endswith(
            ".pdf"
        ):
            filename += ".pdf"

        h, w = self.scanned_image.shape[:2]

        # Temporary high-quality JPEG
        temp_file = os.path.join(
            os.path.dirname(filename),
            "_scanpro_temp.jpg"
        )

        cv2.imwrite(
            temp_file,
            self.scanned_image,
            [
                cv2.IMWRITE_JPEG_QUALITY,
                98
            ]
        )

        # A4 size in points
        page_w, page_h = A4

        ratio = min(
            page_w / w,
            page_h / h
        )

        draw_w = w * ratio
        draw_h = h * ratio

        x = (
            page_w - draw_w
        ) / 2

        y = (
            page_h - draw_h
        ) / 2

        pdf = canvas.Canvas(
            filename,
            pagesize=A4
        )

        pdf.drawImage(
            ImageReader(temp_file),
            x,
            y,
            width=draw_w,
            height=draw_h,
            preserveAspectRatio=True
        )

        pdf.showPage()
        pdf.save()

        try:
            os.remove(
                temp_file
            )
        except:
            pass

        QMessageBox.information(
            self,
            "PDF",
            "PDF exported successfully."
        )


# ============================================================
# APPLICATION
# ============================================================

if __name__ == "__main__":

    app = QApplication(
        sys.argv
    )

    app.setStyle(
        "Fusion"
    )

    window = ScannerApp()

    window.show()

    sys.exit(
        app.exec()
    )
