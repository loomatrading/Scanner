import sys
import os
import cv2
import numpy as np

from PySide6.QtCore import Qt, QPointF, Signal, QTimer, QSize
from PySide6.QtGui import QImage, QPixmap, QPainter, QPen, QBrush, QColor, QFont
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton,
    QFileDialog, QMessageBox, QVBoxLayout, QHBoxLayout, QSlider,
    QComboBox, QFrame, QListWidget, QListWidgetItem, QScrollArea,
    QSizePolicy
)

from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader


# ============================================================
# IMAGE FUNCTIONS
# ============================================================

def order_points(points):
    pts = np.asarray(points, dtype=np.float32)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).reshape(-1)
    return np.array([
        pts[np.argmin(s)],   # TL
        pts[np.argmin(d)],   # TR
        pts[np.argmax(s)],   # BR
        pts[np.argmax(d)]    # BL
    ], dtype=np.float32)


def detect_document_corners(image):
    h, w = image.shape[:2]
    scale = min(1.0, 1400.0 / max(h, w))
    small = cv2.resize(image, None, fx=scale, fy=scale,
                       interpolation=cv2.INTER_AREA) if scale < 1 else image.copy()

    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    candidates = []

    # Bright/white page detection
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    th = cv2.morphologyEx(
        th, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9)), iterations=2
    )
    c, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates.extend(c)

    # Edge detection
    edges = cv2.Canny(gray, 30, 120)
    edges = cv2.morphologyEx(
        edges, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8), iterations=2
    )
    c, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates.extend(c)

    image_area = small.shape[0] * small.shape[1]
    best = None
    best_score = -1

    for contour in candidates:
        area = cv2.contourArea(contour)
        if area < image_area * 0.12:
            continue

        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.025 * peri, True)

        if len(approx) != 4 or not cv2.isContourConvex(approx):
            continue

        pts = approx.reshape(4, 2).astype(np.float32)
        rect_area = cv2.contourArea(pts)
        if rect_area <= 0:
            continue

        rectangularity = area / rect_area
        if rectangularity < 0.72:
            continue

        score = (area / image_area) * 2.2 + rectangularity

        if score > best_score:
            best_score = score
            best = pts

    if best is not None:
        if scale < 1:
            best /= scale
        return order_points(best)

    mx, my = w * .04, h * .04
    return np.array([
        [mx, my], [w - mx, my],
        [w - mx, h - my], [mx, h - my]
    ], dtype=np.float32)


def perspective_transform(image, corners):
    pts = order_points(corners)
    tl, tr, br, bl = pts

    width = int(max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl)))
    height = int(max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr)))

    width = max(width, 300)
    height = max(height, 300)

    dst = np.array([
        [0, 0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0, height - 1]
    ], dtype=np.float32)

    matrix = cv2.getPerspectiveTransform(pts, dst)
    return cv2.warpPerspective(
        image, matrix, (width, height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE
    )


def gray_normalize(gray):
    bg = cv2.GaussianBlur(gray, (0, 0), 25)
    bg = np.maximum(bg, 1)
    out = cv2.divide(gray, bg, scale=255)
    return cv2.normalize(out, None, 0, 255, cv2.NORM_MINMAX)


def document_enhance(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    result = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

    gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    corrected = gray_normalize(gray)

    hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV)
    hsv[:, :, 2] = corrected
    result = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    blur = cv2.GaussianBlur(result, (0, 0), 1.1)
    return cv2.addWeighted(result, 1.16, blur, -0.16, 0)


def bw_enhance(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = gray_normalize(gray)
    clahe = cv2.createCLAHE(clipLimit=2.4, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    out = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 31, 10
    )
    out = cv2.morphologyEx(out, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    return cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)


def color_enhance(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=1.8, tileGridSize=(8, 8)).apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)


def process_document(img, mode, brightness=0, contrast=100, sharpness=20):
    if mode == "Original":
        result = img.copy()
    elif mode == "B&W":
        result = bw_enhance(img)
    elif mode == "Color":
        result = color_enhance(img)
    elif mode == "Lighten":
        result = document_enhance(img)
        result = cv2.convertScaleAbs(result, alpha=1.0, beta=12)
    elif mode == "Magic Pro":
        result = document_enhance(img)
        result = cv2.convertScaleAbs(result, alpha=1.04, beta=4)
    else:  # Enhance
        result = document_enhance(img)

    if brightness:
        result = cv2.convertScaleAbs(result, alpha=1.0, beta=int(brightness))

    alpha = max(0.5, float(contrast) / 100.0)
    result = cv2.convertScaleAbs(result, alpha=alpha, beta=0)

    if sharpness > 0:
        amount = sharpness / 100.0
        blur = cv2.GaussianBlur(result, (0, 0), 1.15)
        result = cv2.addWeighted(
            result, 1.0 + amount * .55, blur, -amount * .55, 0
        )

    return result


def cv_to_pixmap(img):
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    qimg = QImage(
        rgb.data, w, h, ch * w, QImage.Format_RGB888
    ).copy()
    return QPixmap.fromImage(qimg)


# ============================================================
# CROP VIEW
# ============================================================

class CropView(QWidget):
    cornersChanged = Signal()

    def __init__(self):
        super().__init__()
        self.image = None
        self.pixmap = None
        self.corners = []
        self.dragging = -1
        self.setMinimumSize(400, 400)
        self.setMouseTracking(True)

    def set_image(self, image):
        self.image = image
        self.pixmap = cv_to_pixmap(image) if image is not None else None
        self.update()

    def set_corners(self, corners):
        self.corners = [
            QPointF(float(p[0]), float(p[1])) for p in corners
        ]
        self.update()

    def image_to_widget(self, p):
        if self.image is None:
            return QPointF()
        h, w = self.image.shape[:2]
        scale = min((self.width()-30)/w, (self.height()-30)/h)
        dw, dh = w * scale, h * scale
        ox = (self.width() - dw) / 2
        oy = (self.height() - dh) / 2
        return QPointF(ox + p.x()*scale, oy + p.y()*scale)

    def widget_to_image(self, p):
        if self.image is None:
            return QPointF()
        h, w = self.image.shape[:2]
        scale = min((self.width()-30)/w, (self.height()-30)/h)
        dw, dh = w * scale, h * scale
        ox = (self.width() - dw) / 2
        oy = (self.height() - dh) / 2
        x = (p.x()-ox)/scale
        y = (p.y()-oy)/scale
        return QPointF(max(0, min(w-1, x)), max(0, min(h-1, y)))

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.SmoothPixmapTransform)
        p.fillRect(self.rect(), QColor("#171717"))

        if self.pixmap is None:
            p.setPen(QColor("#777777"))
            p.drawText(self.rect(), Qt.AlignCenter, "Import a document")
            return

        h, w = self.image.shape[:2]
        scale = min((self.width()-30)/w, (self.height()-30)/h)
        dw, dh = int(w*scale), int(h*scale)
        x = (self.width()-dw)//2
        y = (self.height()-dh)//2

        shown = self.pixmap.scaled(
            dw, dh, Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        p.drawPixmap(x, y, shown)

        if len(self.corners) == 4:
            pts = [self.image_to_widget(c) for c in self.corners]

            # Darken outside crop area
            overlay = QBrush(QColor(0, 0, 0, 105))
            p.setBrush(overlay)
            p.setPen(Qt.NoPen)

            # Four outside rectangles
            left = min(q.x() for q in pts)
            right = max(q.x() for q in pts)
            top = min(q.y() for q in pts)
            bottom = max(q.y() for q in pts)
            p.drawRect(0, 0, self.width(), max(0, int(top)))
            p.drawRect(0, int(bottom), self.width(), self.height()-int(bottom))
            p.drawRect(0, int(top), max(0, int(left)), int(bottom-top))
            p.drawRect(int(right), int(top), self.width()-int(right), int(bottom-top))

            # Green crop border
            p.setBrush(Qt.NoBrush)
            p.setPen(QPen(QColor("#20C997"), 3))
            for i in range(4):
                p.drawLine(pts[i], pts[(i+1) % 4])

            # Corner handles
            p.setBrush(QBrush(QColor("#20C997")))
            p.setPen(QPen(QColor("#FFFFFF"), 2))
            for q in pts:
                p.drawEllipse(q, 9, 9)

    def mousePressEvent(self, e):
        if len(self.corners) != 4:
            return
        pos = e.position()
        for i, c in enumerate(self.corners):
            q = self.image_to_widget(c)
            if ((q.x()-pos.x())**2 + (q.y()-pos.y())**2) ** .5 <= 28:
                self.dragging = i
                return

    def mouseMoveEvent(self, e):
        if self.dragging < 0:
            return
        self.corners[self.dragging] = self.widget_to_image(e.position())
        self.update()
        self.cornersChanged.emit()

    def mouseReleaseEvent(self, e):
        self.dragging = -1


# ============================================================
# RESULT VIEW
# ============================================================

class ResultView(QLabel):
    def __init__(self):
        super().__init__()
        self.pix = None
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("background:#F1F1F1;")
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    def set_image(self, image):
        self.pix = cv_to_pixmap(image)
        self.refresh()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.refresh()

    def refresh(self):
        if self.pix is not None:
            self.setPixmap(self.pix.scaled(
                self.size()-QSize(30, 30),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            ))


# ============================================================
# MAIN WINDOW
# ============================================================

class ScanPro(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SCAN PRO")
        self.resize(1500, 900)
        self.setMinimumSize(1100, 700)

        self.original = None
        self.scanned = None
        self.pages = []
        self.current_page = -1
        self.mode = "Magic Pro"

        self.timer = QTimer()
        self.timer.setSingleShot(True)
        self.timer.timeout.connect(self.process)

        self.build_ui()

    # --------------------------------------------------------
    # STYLE
    # --------------------------------------------------------

    def css(self):
        return """
        QMainWindow { background:#FFFFFF; }
        QWidget { font-family:"Segoe UI"; }

        #topbar {
            background:#FFFFFF;
            border-bottom:1px solid #E4E4E4;
        }

        #brand {
            color:#202020;
            font-size:23px;
            font-weight:700;
        }

        #thumbPanel {
            background:#FAFAFA;
            border-right:1px solid #E3E3E3;
        }

        #workPanel {
            background:#F1F1F1;
        }

        #tools {
            background:#FFFFFF;
            border-left:1px solid #E3E3E3;
        }

        QPushButton {
            border:none;
            border-radius:8px;
            background:#F4F4F4;
            color:#333333;
        }

        QPushButton:hover {
            background:#E9E9E9;
        }

        QPushButton[selected="true"] {
            border:2px solid #18BFA4;
            background:#E9FBF7;
        }

        #toolButton {
            min-width:82px;
            min-height:76px;
            font-size:13px;
        }

        #cropButton {
            min-height:48px;
            font-size:16px;
        }

        #confirm {
            background:#10B995;
            color:white;
            font-size:16px;
            font-weight:600;
            border-radius:8px;
            min-height:48px;
        }

        #confirm:hover {
            background:#0EAA8B;
        }

        #thumb {
            background:white;
            border:1px solid #DDDDDD;
            border-radius:7px;
        }

        #thumb:hover {
            border:2px solid #21BFA6;
        }

        #status {
            color:#777777;
            font-size:12px;
        }

        QComboBox {
            border:1px solid #DDDDDD;
            border-radius:7px;
            padding:7px;
            background:white;
        }

        QSlider::groove:horizontal {
            height:4px;
            background:#DDDDDD;
            border-radius:2px;
        }

        QSlider::handle:horizontal {
            width:14px;
            margin:-5px 0;
            border-radius:7px;
            background:#18BFA4;
        }
        """

    # --------------------------------------------------------
    # UI
    # --------------------------------------------------------

    def build_ui(self):
        self.setStyleSheet(self.css())

        root = QWidget()
        self.setCentralWidget(root)
        main = QVBoxLayout(root)
        main.setContentsMargins(0,0,0,0)
        main.setSpacing(0)

        # Top bar
        top = QFrame()
        top.setObjectName("topbar")
        top.setFixedHeight(72)
        tl = QHBoxLayout(top)
        tl.setContentsMargins(22, 0, 18, 0)

        brand = QLabel("SCAN PRO")
        brand.setObjectName("brand")
        tl.addWidget(brand)

        sub = QLabel("  Document Scanner")
        sub.setStyleSheet("color:#888888;")
        tl.addWidget(sub)
        tl.addStretch()

        self.size_label = QLabel("")
        self.size_label.setObjectName("status")
        tl.addWidget(self.size_label)

        close = QPushButton("✕")
        close.setFixedSize(40,40)
        close.clicked.connect(self.close)
        tl.addWidget(close)

        main.addWidget(top)

        body = QHBoxLayout()
        body.setContentsMargins(0,0,0,0)
        body.setSpacing(0)

        # Left thumbnails
        thumbs = QFrame()
        thumbs.setObjectName("thumbPanel")
        thumbs.setFixedWidth(190)
        lv = QVBoxLayout(thumbs)
        lv.setContentsMargins(15,20,15,15)

        title = QLabel("Pages")
        title.setStyleSheet("font-weight:600;color:#555;")
        lv.addWidget(title)

        self.thumb_list = QListWidget()
        self.thumb_list.setStyleSheet("""
            QListWidget { border:none;background:transparent; }
            QListWidget::item { margin-bottom:12px; }
            QListWidget::item:selected { background:transparent; }
        """)
        self.thumb_list.currentRowChanged.connect(self.select_page)
        lv.addWidget(self.thumb_list)

        add = QPushButton("＋")
        add.setFixedHeight(58)
        add.setStyleSheet("""
            QPushButton { background:#F0F0F0;font-size:28px;color:#888; }
            QPushButton:hover { background:#E4E4E4; }
        """)
        add.clicked.connect(self.import_file)
        lv.addWidget(add)

        body.addWidget(thumbs)

        # Center
        center = QFrame()
        center.setObjectName("workPanel")
        cl = QVBoxLayout(center)
        cl.setContentsMargins(24,20,24,20)

        self.crop_view = CropView()
        self.crop_view.cornersChanged.connect(self.schedule_process)
        cl.addWidget(self.crop_view, 1)

        body.addWidget(center, 1)

        # Right tools
        tools = QFrame()
        tools.setObjectName("tools")
        tools.setFixedWidth(245)
        tv = QVBoxLayout(tools)
        tv.setContentsMargins(18,20,18,15)
        tv.setSpacing(12)

        rot = QHBoxLayout()
        for text, direction in [("↶", -1), ("↷", 1), ("▣", 0), ("▱", 0)]:
            b = QPushButton(text)
            b.setFixedSize(50,45)
            if direction:
                b.clicked.connect(lambda checked=False, d=direction: self.rotate(d))
            rot.addWidget(b)
        tv.addLayout(rot)

        crop = QPushButton("⌗   Crop")
        crop.setObjectName("cropButton")
        crop.clicked.connect(self.auto_crop)
        tv.addWidget(crop)

        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setStyleSheet("color:#E6E6E6;")
        tv.addWidget(line)

        grid = QHBoxLayout()
        self.tool_buttons = {}

        for name, caption in [
            ("Original", "▤\nOriginal"),
            ("Enhance", "▰\nEnhance"),
            ("Magic Pro", "A⁺\nMagic Pro"),
            ("Lighten", "☀\nLighten"),
            ("B&W", "◐\nB&W"),
            ("Color", "◈\nColor")
        ]:
            b = QPushButton(caption)
            b.setObjectName("toolButton")
            b.setProperty("selected", name == "Magic Pro")
            b.clicked.connect(lambda checked=False, n=name: self.set_mode(n))
            self.tool_buttons[name] = b
            grid.addWidget(b)
            if grid.count() == 2:
                tv.addLayout(grid)
                grid = QHBoxLayout()

        if grid.count():
            tv.addLayout(grid)

        # Fine controls
        fine = QLabel("Fine adjustment")
        fine.setStyleSheet("font-weight:600;color:#555;margin-top:8px;")
        tv.addWidget(fine)

        self.brightness = QSlider(Qt.Horizontal)
        self.brightness.setRange(-25, 25)
        self.brightness.setValue(4)
        self.brightness.valueChanged.connect(self.schedule_process)
        tv.addWidget(QLabel("Brightness"))
        tv.addWidget(self.brightness)

        self.contrast = QSlider(Qt.Horizontal)
        self.contrast.setRange(80, 125)
        self.contrast.setValue(105)
        self.contrast.valueChanged.connect(self.schedule_process)
        tv.addWidget(QLabel("Contrast"))
        tv.addWidget(self.contrast)

        self.sharpness = QSlider(Qt.Horizontal)
        self.sharpness.setRange(0, 100)
        self.sharpness.setValue(25)
        self.sharpness.valueChanged.connect(self.schedule_process)
        tv.addWidget(QLabel("Sharpness"))
        tv.addWidget(self.sharpness)

        tv.addStretch()

        self.confirm = QPushButton("Confirm")
        self.confirm.setObjectName("confirm")
        self.confirm.clicked.connect(self.save_scan)
        tv.addWidget(self.confirm)

        body.addWidget(tools)
        main.addLayout(body, 1)

        # Bottom status
        bottom = QFrame()
        bottom.setFixedHeight(36)
        bottom.setStyleSheet("background:#FFFFFF;border-top:1px solid #E5E5E5;")
        bl = QHBoxLayout(bottom)
        bl.setContentsMargins(20,0,20,0)
        self.status = QLabel("Import a document to begin")
        self.status.setObjectName("status")
        bl.addWidget(self.status)
        bl.addStretch()
        pdf = QPushButton("Export PDF")
        pdf.clicked.connect(self.export_pdf)
        bl.addWidget(pdf)
        main.addWidget(bottom)

    # --------------------------------------------------------
    # PAGES
    # --------------------------------------------------------

    def import_file(self):
        fn, _ = QFileDialog.getOpenFileName(
            self, "Import Document", "",
            "Images (*.jpg *.jpeg *.png *.bmp *.tif *.tiff)"
        )
        if not fn:
            return

        image = cv2.imread(fn, cv2.IMREAD_COLOR)
        if image is None:
            QMessageBox.critical(self, "Error", "Could not open the image.")
            return

        corners = detect_document_corners(image)
        self.pages.append({"image": image, "corners": corners, "scan": None})
        self.rebuild_thumbnails()
        self.thumb_list.setCurrentRow(len(self.pages)-1)

    def rebuild_thumbnails(self):
        self.thumb_list.blockSignals(True)
        self.thumb_list.clear()

        for i, page in enumerate(self.pages):
            pix = cv_to_pixmap(page["image"])
            pix = pix.scaled(135, 175, Qt.KeepAspectRatio, Qt.SmoothTransformation)

            label = QLabel()
            label.setObjectName("thumb")
            label.setAlignment(Qt.AlignCenter)
            label.setPixmap(pix)
            label.setFixedSize(150, 190)

            item = QListWidgetItem()
            item.setSizeHint(QSize(155, 200))
            self.thumb_list.addItem(item)
            self.thumb_list.setItemWidget(item, label)

        self.thumb_list.blockSignals(False)

    def select_page(self, row):
        if row < 0 or row >= len(self.pages):
            return
        self.current_page = row
        page = self.pages[row]
        self.original = page["image"]
        self.crop_view.set_image(self.original)
        self.crop_view.set_corners(page["corners"])
        self.process()

    # --------------------------------------------------------
    # PROCESSING
    # --------------------------------------------------------

    def schedule_process(self):
        if self.original is not None:
            self.timer.start(70)

    def process(self):
        if self.original is None or len(self.crop_view.corners) != 4:
            return

        corners = np.array(
            [[p.x(), p.y()] for p in self.crop_view.corners],
            dtype=np.float32
        )

        if self.current_page >= 0:
            self.pages[self.current_page]["corners"] = corners

        crop = perspective_transform(self.original, corners)

        result = process_document(
            crop,
            self.mode,
            self.brightness.value(),
            self.contrast.value(),
            self.sharpness.value()
        )

        self.scanned = result

        if self.current_page >= 0:
            self.pages[self.current_page]["scan"] = result

        h, w = result.shape[:2]
        self.size_label.setText(f"{w} × {h} px")
        self.status.setText("Live scan preview — drag the four corners to refine")

        # The crop view itself remains the editable original.
        # The actual scanned result is shown in a preview dialog
        # only when requested by the user via Confirm.
        # This keeps the main canvas close to the reference UI.

    # --------------------------------------------------------
    # MODE
    # --------------------------------------------------------

    def set_mode(self, name):
        self.mode = name
        for n, b in self.tool_buttons.items():
            b.setProperty("selected", n == name)
            b.style().unpolish(b)
            b.style().polish(b)
        self.schedule_process()

        # Show live processed result in a lightweight overlay window
        # when an enhancement tool is selected.
        self.show_result_preview()

    def show_result_preview(self):
        if self.scanned is None:
            self.process()
        if self.scanned is None:
            return

        # Use a compact non-modal preview only when enhancement is selected.
        # The main crop canvas stays visible and editable.
        if not hasattr(self, "_preview"):
            self._preview = ResultPreview(self)

        self._preview.set_image(self.scanned)
        self._preview.show()
        self._preview.raise_()
        self._preview.activateWindow()

    # --------------------------------------------------------
    # CROP / ROTATE
    # --------------------------------------------------------

    def auto_crop(self):
        if self.original is None:
            return
        corners = detect_document_corners(self.original)
        self.crop_view.set_corners(corners)
        self.schedule_process()

    def rotate(self, direction):
        if self.original is None:
            return

        if direction > 0:
            self.original = cv2.rotate(self.original, cv2.ROTATE_90_CLOCKWISE)
        else:
            self.original = cv2.rotate(self.original, cv2.ROTATE_90_COUNTERCLOCKWISE)

        corners = detect_document_corners(self.original)

        if self.current_page >= 0:
            self.pages[self.current_page]["image"] = self.original
            self.pages[self.current_page]["corners"] = corners

        self.crop_view.set_image(self.original)
        self.crop_view.set_corners(corners)
        self.rebuild_thumbnails()
        self.thumb_list.setCurrentRow(self.current_page)
        self.process()

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    def save_scan(self):
        if self.scanned is None:
            self.process()
        if self.scanned is None:
            QMessageBox.warning(self, "No Scan", "Import a document first.")
            return

        fn, _ = QFileDialog.getSaveFileName(
            self, "Save Scan", "ScanPro.jpg",
            "JPEG (*.jpg *.jpeg);;PNG (*.png)"
        )
        if not fn:
            return

        if fn.lower().endswith(".png"):
            cv2.imwrite(fn, self.scanned, [cv2.IMWRITE_PNG_COMPRESSION, 1])
        else:
            if not fn.lower().endswith((".jpg", ".jpeg")):
                fn += ".jpg"
            cv2.imwrite(fn, self.scanned, [cv2.IMWRITE_JPEG_QUALITY, 98])

        QMessageBox.information(self, "Saved", "Scan saved successfully.")

    def export_pdf(self):
        if not self.pages:
            QMessageBox.warning(self, "No Pages", "Import a document first.")
            return

        fn, _ = QFileDialog.getSaveFileName(
            self, "Export PDF", "ScanPro.pdf", "PDF (*.pdf)"
        )
        if not fn:
            return
        if not fn.lower().endswith(".pdf"):
            fn += ".pdf"

        pdf = canvas.Canvas(fn, pagesize=A4)
        pw, ph = A4

        temp_files = []

        for i, page in enumerate(self.pages):
            if page["scan"] is None:
                corners = page["corners"]
                crop = perspective_transform(page["image"], corners)
                scan = process_document(
                    crop, self.mode,
                    self.brightness.value(),
                    self.contrast.value(),
                    self.sharpness.value()
                )
            else:
                scan = page["scan"]

            temp = os.path.join(
                os.path.dirname(fn), f"_scanpro_{os.getpid()}_{i}.jpg"
            )
            cv2.imwrite(temp, scan, [cv2.IMWRITE_JPEG_QUALITY, 98])
            temp_files.append(temp)

            h, w = scan.shape[:2]
            ratio = min(pw / w, ph / h)
            dw, dh = w * ratio, h * ratio
            x, y = (pw-dw)/2, (ph-dh)/2

            pdf.drawImage(
                ImageReader(temp), x, y,
                width=dw, height=dh,
                preserveAspectRatio=True
            )
            pdf.showPage()

        pdf.save()

        for temp in temp_files:
            try:
                os.remove(temp)
            except OSError:
                pass

        QMessageBox.information(self, "PDF", "PDF exported successfully.")


# ============================================================
# RESULT PREVIEW
# ============================================================

class ResultPreview(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Scan Preview")
        self.resize(650, 820)
        self.setWindowFlags(
            Qt.Window | Qt.WindowStaysOnTopHint
        )
        self.setStyleSheet("""
            QFrame { background:#FFFFFF; }
            QLabel { background:#F1F1F1; }
            QPushButton {
                background:#10B995;color:white;border:none;
                border-radius:7px;padding:10px 22px;
            }
        """)

        lay = QVBoxLayout(self)
        self.image = QLabel()
        self.image.setAlignment(Qt.AlignCenter)
        lay.addWidget(self.image, 1)

        close = QPushButton("Close Preview")
        close.clicked.connect(self.hide)
        lay.addWidget(close)

    def set_image(self, image):
        self.pix = cv_to_pixmap(image)
        self.refresh()

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self.refresh()

    def refresh(self):
        if hasattr(self, "pix") and self.pix:
            self.image.setPixmap(
                self.pix.scaled(
                    self.image.size()-QSize(20,20),
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation
                )
            )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = ScanPro()
    win.show()
    sys.exit(app.exec())
