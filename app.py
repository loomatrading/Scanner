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


def ai_enhance_document(img):
    """Fast offline AI-style document enhancement using OpenCV."""
    if img is None:
        return None

    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
    result = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)

    # Moderate denoise + detail recovery.
    result = cv2.fastNlMeansDenoisingColored(result, None, 3, 3, 7, 21)
    blur = cv2.GaussianBlur(result, (0, 0), 1.0)
    result = cv2.addWeighted(result, 1.15, blur, -0.15, 0)

    # Correct uneven illumination on paper.
    gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    bg = cv2.GaussianBlur(gray, (0, 0), 25)
    bg = np.maximum(bg, 1)
    normalized = cv2.divide(gray, bg, scale=255)

    hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV)
    hsv[:, :, 2] = cv2.normalize(normalized, None, 0, 255, cv2.NORM_MINMAX)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def remove_document_background(img):
    """Offline page-background cleanup; no cloud/API is required."""
    if img is None:
        return None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    bg = cv2.GaussianBlur(gray, (0, 0), 21)
    diff = cv2.absdiff(gray, bg)

    mask = np.where((gray > 190) & (diff < 18), 255, 0).astype(np.uint8)
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    )
    mask = cv2.GaussianBlur(mask, (0, 0), 1.2)

    white = np.full_like(img, 255)
    alpha = (mask.astype(np.float32) / 255.0)[:, :, None]
    return (img.astype(np.float32) * (1 - alpha) +
            white.astype(np.float32) * alpha).astype(np.uint8)


def ai_magic_pro(img):
    return remove_document_background(ai_enhance_document(img))


def noise_reduction(img):
    if img is None:
        return None
    return cv2.fastNlMeansDenoisingColored(img, None, 5, 5, 7, 21)


def process_document(img, mode, brightness=0, contrast=100, sharpness=20):
    if img is None:
        return None

    if mode == "Original":
        result = img.copy()
    elif mode == "B&W":
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        gray = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8)).apply(gray)
        result = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 31, 9
        )
        result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
    elif mode == "Color":
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = cv2.createCLAHE(clipLimit=1.8, tileGridSize=(8, 8)).apply(l)
        result = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)
    elif mode == "Lighten":
        result = cv2.convertScaleAbs(ai_enhance_document(img), alpha=1.0, beta=12)
    elif mode == "Magic Pro":
        result = ai_magic_pro(img)
    else:
        result = ai_enhance_document(img)

    if brightness:
        result = cv2.convertScaleAbs(result, alpha=1.0, beta=int(brightness))

    alpha = max(0.5, float(contrast) / 100.0)
    if abs(alpha - 1.0) > 0.001:
        result = cv2.convertScaleAbs(result, alpha=alpha, beta=0)

    if sharpness > 0 and mode != "B&W":
        amount = sharpness / 100.0
        blur = cv2.GaussianBlur(result, (0, 0), 1.0)
        result = cv2.addWeighted(
            result, 1.0 + amount * .45, blur, -amount * .45, 0
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
        p.fillRect(self.rect(), QColor("#FFFFFF"))

        if self.pixmap is None:
            p.setPen(QColor("#222222"))
            p.drawText(self.rect(), Qt.AlignCenter, "Import Images")
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
        self.setWindowTitle("Scan Pro")
        self.resize(1500, 900)
        self.setMinimumSize(1120, 720)

        self.original = None
        self.scanned = None
        self.pages = []
        self.current_page = -1
        self.mode = "Magic Pro"
        self._preview = None

        self.timer = QTimer(self)
        self.timer.setSingleShot(True)
        self.timer.timeout.connect(self.process)

        self.build_ui()

    # --------------------------------------------------------
    # Fast inline SVG icons: no external icon/image files.
    # --------------------------------------------------------

    def icon_svg(self, kind, color):
        data = {
            "open": f'<path d="M5 12h12l4 4h14v17H5z" fill="{color}" opacity=".18"/><path d="M5 12h12l4 4h14v17H5zM7 18h25" fill="none" stroke="{color}" stroke-width="2.5"/>',
            "rotate": f'<path d="M13 13a12 12 0 1 1-2 17" fill="none" stroke="{color}" stroke-width="3"/><path d="M7 13l7-1-2 7" fill="none" stroke="{color}" stroke-width="3"/>',
            "crop": f'<path d="M10 6v27h27M6 10h27v27" fill="none" stroke="{color}" stroke-width="3"/>',
            "original": f'<rect x="8" y="5" width="28" height="38" rx="3" fill="{color}" opacity=".15"/><rect x="8" y="5" width="28" height="38" rx="3" fill="none" stroke="{color}" stroke-width="2.5"/><path d="M14 15h16M14 22h16M14 29h13M14 36h8" stroke="{color}" stroke-width="3"/>',
            "enhance": f'<path d="M8 37l8-16 7 8 7-15 7 23z" fill="{color}" opacity=".9"/><circle cx="30" cy="11" r="5" fill="{color}" opacity=".35"/>',
            "ai": '<text x="3" y="38" font-family="Arial" font-size="31" font-weight="700" fill="#00B89C">AI</text><path d="M39 7l2 6 6 2-6 2-2 6-2-6-6-2 6-2z" fill="#18C9A7"/>',
            "light": f'<circle cx="24" cy="24" r="10" fill="{color}"/><path d="M24 4v7M24 37v7M4 24h7M37 24h7M10 10l5 5M33 33l5 5M38 10l-5 5M15 33l-5 5" stroke="{color}" stroke-width="3"/>',
            "bw": '<circle cx="19" cy="24" r="12" fill="#111"/><circle cx="29" cy="24" r="12" fill="#fff" stroke="#777" stroke-width="1"/>',
            "color": '<circle cx="24" cy="24" r="17" fill="#F4B942" opacity=".35"/><circle cx="24" cy="24" r="12" fill="#EF5DA8" opacity=".55"/><circle cx="24" cy="24" r="7" fill="#55C7E8"/>',
            "magic": '<path d="M8 33L30 10l6 6L14 39z" fill="#7C3AED" opacity=".18"/><path d="M9 31L29 11l5 5L14 36z" fill="none" stroke="#7C3AED" stroke-width="2.5"/><path d="M33 5l2 5 5 2-5 2-2 5-2-5-5-2 5-2z" fill="#18C9A7"/>',
            "removebg": '<path d="M8 8h12v6h-6v14h14v-6h6v12H8z" fill="#2F80ED" opacity=".15"/><path d="M8 8h12v6h-6v14h14v-6h6v12H8z" fill="none" stroke="#2F80ED" stroke-width="2.5"/>',
            "straight": '<rect x="8" y="10" width="25" height="22" rx="2" fill="#3B82F6" opacity=".12" transform="rotate(-6 20 21)"/><rect x="8" y="10" width="25" height="22" rx="2" fill="none" stroke="#3B82F6" stroke-width="2.5" transform="rotate(-6 20 21)"/>',
            "noise": '<circle cx="12" cy="14" r="3" fill="#EF4444"/><circle cx="23" cy="9" r="2" fill="#F97316"/><circle cx="30" cy="19" r="3" fill="#EF4444"/><circle cx="15" cy="28" r="2" fill="#F97316"/><path d="M8 35c8-8 17-8 25 0" fill="none" stroke="#EF4444" stroke-width="2"/>'
        }
        return f'<svg xmlns="http://www.w3.org/2000/svg" width="48" height="48" viewBox="0 0 48 48">{data.get(kind, "")}</svg>'

    def make_icon(self, kind, color="#1677FF", size=44):
        from PySide6.QtSvg import QSvgRenderer
        pix = QPixmap(size, size)
        pix.fill(Qt.transparent)
        painter = QPainter(pix)
        renderer = QSvgRenderer(self.icon_svg(kind, color).encode())
        renderer.render(painter)
        painter.end()
        from PySide6.QtGui import QIcon
        return QIcon(pix)

    # --------------------------------------------------------
    # STYLE
    # --------------------------------------------------------

    def css(self):
        return """
        QMainWindow { background:#FFFFFF; }
        QWidget { font-family:"Segoe UI"; color:#202020; }

        #topbar { background:#FFFFFF; border-bottom:1px solid #E5E7EB; }
        #brand { color:#202020; font-size:20px; font-weight:700; }
        #thumbPanel { background:#FFFFFF; border-right:1px solid #E3E5E8; }
        #workPanel { background:#F1F2F4; }
        #canvasCard { background:#FFFFFF; border-radius:10px; }
        #tools { background:#FFFFFF; border-left:1px solid #E3E5E8; }

        #toolButton {
            background:#F2F2F2; border-radius:9px;
            min-width:82px; min-height:72px;
            font-size:13px; color:#252525;
        }
        #toolButton:hover { background:#EAECEE; }
        #toolButton[selected="true"] {
            background:#E8FBF7; border:1.5px solid #12BFA3; color:#00B89C;
        }

        #topButton { background:#F4F4F4; border-radius:8px; }
        #topButton:hover { background:#EAEAEA; }

        #aiPanel {
            background:#FFFFFF; border:1px solid #E1E4E7; border-radius:12px;
        }
        #aiItem {
            background:#FFFFFF; border-radius:8px;
            text-align:left; padding:4px;
        }
        #aiItem:hover { background:#F4F7F9; }

        #bottomBar { background:#FFFFFF; border-top:1px solid #E5E7EB; }
        #saveButton {
            background:#10B99A; color:white; border-radius:8px;
            font-size:20px; font-weight:600;
            min-width:155px; min-height:45px;
        }
        #saveButton:hover { background:#0DAE91; }
        #status { color:#777; font-size:12px; }

        QComboBox {
            border:none; background:transparent; font-size:15px; padding:5px;
        }
        QSlider::groove:horizontal {
            height:4px; background:#DDDDDD; border-radius:2px;
        }
        QSlider::handle:horizontal {
            width:14px; margin:-5px 0; border-radius:7px; background:#18BFA4;
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
        main.setContentsMargins(0, 0, 0, 0)
        main.setSpacing(0)

        top = QFrame()
        top.setObjectName("topbar")
        top.setFixedHeight(50)
        tl = QHBoxLayout(top)
        tl.setContentsMargins(18, 0, 18, 0)

        logo = QLabel("▣")
        logo.setStyleSheet("color:#10B99A;font-size:22px;font-weight:bold;")
        tl.addWidget(logo)
        brand = QLabel("Scan Pro")
        brand.setObjectName("brand")
        tl.addWidget(brand)
        tl.addStretch()

        self.size_label = QLabel("")
        self.size_label.setObjectName("status")
        tl.addWidget(self.size_label)

        close = QPushButton("✕")
        close.setFixedSize(34, 34)
        close.clicked.connect(self.close)
        tl.addWidget(close)
        main.addWidget(top)

        body = QHBoxLayout()
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(0)

        # LEFT: pages
        thumbs = QFrame()
        thumbs.setObjectName("thumbPanel")
        thumbs.setFixedWidth(300)
        lv = QVBoxLayout(thumbs)
        lv.setContentsMargins(24, 20, 18, 15)

        self.thumb_list = QListWidget()
        self.thumb_list.setStyleSheet("""
            QListWidget { border:none; background:transparent; }
            QListWidget::item { margin-bottom:10px; }
            QListWidget::item:selected { background:#E8FBF7; border-radius:10px; }
        """)
        self.thumb_list.currentRowChanged.connect(self.select_page)
        lv.addWidget(self.thumb_list, 1)

        add = QPushButton("+")
        add.setFixedSize(54, 54)
        add.setStyleSheet(
            "QPushButton{background:#F0F0F0;color:#999;font-size:35px;border-radius:8px;}"
            "QPushButton:hover{background:#E5E5E5;}"
        )
        add.clicked.connect(self.import_file)
        lv.addWidget(add, 0, Qt.AlignHCenter)
        body.addWidget(thumbs)

        # CENTER
        center = QFrame()
        center.setObjectName("workPanel")
        cl = QVBoxLayout(center)
        cl.setContentsMargins(24, 14, 24, 14)

        self.canvas = QFrame()
        self.canvas.setObjectName("canvasCard")
        canvas_lay = QVBoxLayout(self.canvas)
        canvas_lay.setContentsMargins(0, 0, 0, 0)

        self.crop_view = CropView()
        self.crop_view.cornersChanged.connect(self.schedule_process)
        canvas_lay.addWidget(self.crop_view)
        cl.addWidget(self.canvas, 1)
        body.addWidget(center, 1)

        # RIGHT
        tools = QFrame()
        tools.setObjectName("tools")
        tools.setFixedWidth(360)
        tv = QVBoxLayout(tools)
        tv.setContentsMargins(20, 24, 20, 14)
        tv.setSpacing(10)

        top_tools = QHBoxLayout()
        for kind, caption, slot in [
            ("open", "Open", self.import_file),
            ("rotate", "Rotate", lambda: self.rotate(1)),
            ("crop", "Crop", self.auto_crop)
        ]:
            holder = QVBoxLayout()
            b = QPushButton()
            b.setObjectName("topButton")
            b.setFixedSize(72, 54)
            b.setIcon(self.make_icon(kind, "#111111", 36))
            b.setIconSize(QSize(36, 36))
            b.clicked.connect(slot)
            holder.addWidget(b, 0, Qt.AlignHCenter)
            lab = QLabel(caption)
            lab.setAlignment(Qt.AlignCenter)
            holder.addWidget(lab)
            top_tools.addLayout(holder)
        tv.addLayout(top_tools)

        grid = QHBoxLayout()
        self.tool_buttons = {}
        defs = [
            ("Original", "Original", "original", "#1677FF"),
            ("Enhance", "Enhance", "enhance", "#2677E8"),
            ("Magic Pro", "Magic Pro AI", "ai", "#00B89C"),
            ("Lighten", "Lighten", "light", "#FFB000"),
            ("B&W", "B&W", "bw", "#111111"),
            ("Color", "Color", "color", "#E6509D")
        ]

        for idx, (name, caption, kind, color) in enumerate(defs):
            holder = QVBoxLayout()
            b = QPushButton()
            b.setObjectName("toolButton")
            b.setProperty("selected", name == "Magic Pro")
            b.setIcon(self.make_icon(kind, color))
            b.setIconSize(QSize(46, 46))
            b.clicked.connect(lambda checked=False, n=name: self.set_mode(n))
            self.tool_buttons[name] = b
            holder.addWidget(b)
            lab = QLabel(caption)
            lab.setAlignment(Qt.AlignCenter)
            holder.addWidget(lab)
            grid.addLayout(holder)

            if idx % 2 == 1:
                tv.addLayout(grid)
                grid = QHBoxLayout()

        if grid.count():
            tv.addLayout(grid)

        ai_panel = QFrame()
        ai_panel.setObjectName("aiPanel")
        av = QVBoxLayout(ai_panel)
        av.setContentsMargins(12, 9, 12, 9)
        av.setSpacing(3)

        head = QHBoxLayout()
        title = QLabel("AI Tools")
        title.setStyleSheet("font-size:16px;font-weight:600;")
        head.addWidget(title)
        head.addStretch()
        new = QLabel("NEW")
        new.setStyleSheet(
            "background:#10B99A;color:white;border-radius:5px;"
            "padding:2px 6px;font-size:10px;font-weight:700;"
        )
        head.addWidget(new)
        av.addLayout(head)

        self.add_ai_button(av, "magic", "AI Enhance", "Improve clarity and details", self.ai_enhance)
        self.add_ai_button(av, "removebg", "Background Removal", "Clean page background", self.ai_background)
        self.add_ai_button(av, "straight", "Auto Straighten", "Detect and straighten document", self.auto_crop)
        self.add_ai_button(av, "noise", "Noise Reduction", "Reduce grain and noise", self.ai_noise)

        tv.addWidget(ai_panel)
        tv.addStretch()

        fine = QLabel("Fine adjustment")
        fine.setStyleSheet("font-weight:600;color:#555;")
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

        body.addWidget(tools)
        main.addLayout(body, 1)

        # BOTTOM: image saving only, default Desktop
        bottom = QFrame()
        bottom.setObjectName("bottomBar")
        bottom.setFixedHeight(70)
        bl = QHBoxLayout(bottom)
        bl.setContentsMargins(34, 10, 24, 10)

        save_type = QLabel("Save as Image")
        save_type.setStyleSheet("font-size:15px;")
        bl.addWidget(save_type)

        self.status = QLabel("Ready")
        self.status.setObjectName("status")
        bl.addWidget(self.status)
        bl.addStretch()

        dest = QLabel("Save to:")
        dest.setStyleSheet("font-size:15px;")
        bl.addWidget(dest)

        self.destination = QComboBox()
        self.destination.addItem("Desktop")
        self.destination.setFixedWidth(130)
        bl.addWidget(self.destination)

        self.save_button = QPushButton("Save")
        self.save_button.setObjectName("saveButton")
        self.save_button.clicked.connect(self.save_scan)
        bl.addWidget(self.save_button)
        main.addWidget(bottom)

    def add_ai_button(self, layout, kind, title, subtitle, slot):
        b = QPushButton(f"   {title}\n   {subtitle}")
        b.setObjectName("aiItem")
        b.setIcon(self.make_icon(kind, size=36))
        b.setIconSize(QSize(36, 36))
        b.setStyleSheet(
            "QPushButton{text-align:left;background:#FFF;border-radius:8px;"
            "padding:5px;font-size:13px;}"
            "QPushButton:hover{background:#F4F7F9;}"
        )
        b.clicked.connect(slot)
        layout.addWidget(b)

    # --------------------------------------------------------
    # FILES
    # --------------------------------------------------------

    def desktop_path(self):
        return os.path.join(os.path.expanduser("~"), "Desktop")

    def import_file(self):
        desktop = self.desktop_path()
        os.makedirs(desktop, exist_ok=True)

        fn, _ = QFileDialog.getOpenFileName(
            self, "Import Images", desktop,
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
        self.thumb_list.setCurrentRow(len(self.pages) - 1)

    def rebuild_thumbnails(self):
        self.thumb_list.blockSignals(True)
        self.thumb_list.clear()

        for i, page in enumerate(self.pages):
            pix = cv_to_pixmap(page["image"])
            pix = pix.scaled(205, 265, Qt.KeepAspectRatio, Qt.SmoothTransformation)

            label = QLabel()
            label.setAlignment(Qt.AlignCenter)
            label.setPixmap(pix)
            label.setFixedSize(250, 285)
            label.setStyleSheet(
                "background:#FFF;border:1px solid #DDE3E5;border-radius:8px;"
            )

            item = QListWidgetItem()
            item.setSizeHint(QSize(260, 300))
            self.thumb_list.addItem(item)
            self.thumb_list.setItemWidget(item, label)

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
            self.timer.start(120)

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
            crop, self.mode,
            self.brightness.value(),
            self.contrast.value(),
            self.sharpness.value()
        )

        self.scanned = result

        if self.current_page >= 0:
            self.pages[self.current_page]["scan"] = result

        h, w = result.shape[:2]
        self.size_label.setText(f"{w} × {h} px")
        self.status.setText("Ready")

    # --------------------------------------------------------
    # AI
    # --------------------------------------------------------

    def show_result(self, result, message):
        if result is None:
            return
        self.scanned = result
        if self.current_page >= 0:
            self.pages[self.current_page]["scan"] = result

        if self._preview is None:
            self._preview = ResultPreview(self)
        self._preview.set_image(result)
        self._preview.show()
        self._preview.raise_()
        self._preview.activateWindow()
        self.status.setText(message)

    def ai_enhance(self):
        if self.original is None:
            QMessageBox.information(self, "AI Enhance", "Import an image first.")
            return
        self.process()
        self.show_result(ai_enhance_document(self.scanned), "AI enhancement applied")

    def ai_background(self):
        if self.original is None:
            QMessageBox.information(self, "Background Removal", "Import an image first.")
            return
        self.process()
        self.show_result(
            remove_document_background(self.scanned),
            "AI background cleanup applied"
        )

    def ai_noise(self):
        if self.original is None:
            QMessageBox.information(self, "Noise Reduction", "Import an image first.")
            return
        self.process()
        self.show_result(noise_reduction(self.scanned), "Noise reduction applied")

    # --------------------------------------------------------
    # MODE / ROTATE / CROP
    # --------------------------------------------------------

    def set_mode(self, name):
        self.mode = name
        for n, b in self.tool_buttons.items():
            b.setProperty("selected", n == name)
            b.style().unpolish(b)
            b.style().polish(b)

        self.process()

        if name == "Magic Pro":
            self.status.setText("Magic Pro AI: enhancement + background cleanup")
        else:
            self.status.setText(f"{name} mode")

        if self.scanned is not None and name != "Original":
            if self._preview is None:
                self._preview = ResultPreview(self)
            self._preview.set_image(self.scanned)
            self._preview.show()
            self._preview.raise_()

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
            self.pages[self.current_page]["scan"] = None

        self.crop_view.set_image(self.original)
        self.crop_view.set_corners(corners)
        self.rebuild_thumbnails()
        self.thumb_list.setCurrentRow(self.current_page)
        self.process()

    # --------------------------------------------------------
    # SAVE IMAGE ONLY
    # --------------------------------------------------------

    def save_scan(self):
        if self.original is None:
            QMessageBox.warning(self, "No Scan", "Import an image first.")
            return

        self.process()
        if self.scanned is None:
            return

        desktop = self.desktop_path()
        os.makedirs(desktop, exist_ok=True)

        default_name = os.path.join(desktop, "ScanPro.jpg")
        fn, _ = QFileDialog.getSaveFileName(
            self, "Save Scan", default_name,
            "JPEG Image (*.jpg *.jpeg);;PNG Image (*.png)"
        )
        if not fn:
            return

        if fn.lower().endswith(".png"):
            cv2.imwrite(fn, self.scanned, [cv2.IMWRITE_PNG_COMPRESSION, 1])
        else:
            if not fn.lower().endswith((".jpg", ".jpeg")):
                fn += ".jpg"
            cv2.imwrite(fn, self.scanned, [cv2.IMWRITE_JPEG_QUALITY, 98])

        self.status.setText(f"Saved: {os.path.basename(fn)}")
        QMessageBox.information(self, "Saved", "Image saved successfully.")

# ============================================================
# RESULT PREVIEW
# ============================================================

class ResultPreview(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Scan Preview")
        self.resize(650, 820)
        self.setWindowFlags(Qt.Window)
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
