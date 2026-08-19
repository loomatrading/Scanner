
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np


class DocumentScanner:
    def _init_(self, root):
        self.root = root
        self.root.title("Document Scanner")
        self.root.geometry("1200x800")
        self.root.minsize(900, 650)

        self.original = None
        self.points = None
        self.display_scale = 1
        self.offset_x = 0
        self.offset_y = 0
        self.drag_point = None
        self.mode = "Original"
        self.tk_image = None

        # =========================
        # Top toolbar
        # =========================
        toolbar = tk.Frame(root, bg="#202124", height=55)
        toolbar.pack(fill="x")

        buttons = [
            ("Import File", self.import_file),
            ("Auto Detect", self.auto_detect),
            ("Reset", self.reset),
            ("Apply Scan", self.apply_scan),
            ("Save JPG", self.save_jpg),
            ("Save PDF", self.save_pdf),
        ]

        for text, command in buttons:
            tk.Button(
                toolbar,
                text=text,
                command=command,
                padx=12,
                pady=7
            ).pack(side="left", padx=5, pady=8)

        # =========================
        # Enhancement toolbar
        # =========================
        enhancement = tk.Frame(root, bg="#eeeeee")
        enhancement.pack(fill="x")

        modes = [
            "Original",
            "Color",
            "Gray",
            "B&W",
            "Magic Scan"
        ]

        for mode in modes:
            tk.Button(
                enhancement,
                text=mode,
                command=lambda m=mode: self.set_mode(m)
            ).pack(side="left", padx=5, pady=5)

        # =========================
        # Canvas
        # =========================
        self.canvas = tk.Canvas(
            root,
            bg="#303030",
            highlightthickness=0
        )
        self.canvas.pack(fill="both", expand=True)

        self.canvas.bind("<Button-1>", self.mouse_down)
        self.canvas.bind("<B1-Motion>", self.mouse_move)
        self.canvas.bind("<ButtonRelease-1>", self.mouse_up)
        self.canvas.bind("<Configure>", lambda event: self.redraw())

        # =========================
        # Status
        # =========================
        self.status = tk.Label(
            root,
            text="Import an image to begin.",
            anchor="w"
        )
        self.status.pack(fill="x")

    # ==========================================================
    # IMPORT FILE
    # ==========================================================

    def import_file(self):

        filename = filedialog.askopenfilename(
            title="Import File",
            filetypes=[
                (
                    "Image Files",
                    "*.jpg *.jpeg *.png *.bmp *.tif *.tiff"
                ),
                ("All Files", ".")
            ]
        )

        if not filename:
            return

        image = cv2.imread(filename)

        if image is None:
            messagebox.showerror(
                "Error",
                "Unable to open this image."
            )
            return

        self.original = image

        # Automatically detect paper
        self.points = self.detect_document(image)

        self.mode = "Original"

        self.status.config(
            text="Image loaded. Drag the blue corners if necessary."
        )

        self.redraw()

    # ==========================================================
    # AUTOMATIC DOCUMENT DETECTION
    # ==========================================================

    def detect_document(self, image):

        height, width = image.shape[:2]

        # Resize for faster detection
        scale = min(
            1.0,
            1200.0 / max(height, width)
        )

        if scale < 1:
            small = cv2.resize(
                image,
                None,
                fx=scale,
                fy=scale
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

        edges = cv2.Canny(
            gray,
            50,
            150
        )

        contours, _ = cv2.findContours(
            edges,
            cv2.RETR_LIST,
            cv2.CHAIN_APPROX_SIMPLE
        )

        contours = sorted(
            contours,
            key=cv2.contourArea,
            reverse=True
        )

        sh, sw = small.shape[:2]

        for contour in contours[:50]:

            perimeter = cv2.arcLength(
                contour,
                True
            )

            approx = cv2.approxPolyDP(
                contour,
                0.02 * perimeter,
                True
            )

            area = cv2.contourArea(approx)

            # Require quadrilateral
            if (
                len(approx) == 4
                and area > 0.15 * sw * sh
            ):

                points = (
                    approx.reshape(4, 2)
                    / scale
                )

                return self.order_points(points)

        # If detection fails,
        # create default rectangle
        margin_x = width * 0.08
        margin_y = height * 0.08

        return np.array(
            [
                [margin_x, margin_y],
                [width - margin_x, margin_y],
                [width - margin_x, height - margin_y],
                [margin_x, height - margin_y]
            ],
            dtype=np.float32
        )

    # ==========================================================
    # ORDER CORNERS
    # ==========================================================

    def order_points(self, points):

        points = np.asarray(
            points,
            dtype=np.float32
        )

        # Top-left has smallest sum
        # Bottom-right has largest sum
        s = points.sum(axis=1)

        # Top-right has smallest difference
        # Bottom-left has largest difference
        d = np.diff(
            points,
            axis=1
        ).ravel()

        return np.array(
            [
                points[np.argmin(s)],   # top-left
                points[np.argmin(d)],   # top-right
                points[np.argmax(s)],   # bottom-right
                points[np.argmax(d)]    # bottom-left
            ],
            dtype=np.float32
        )

    # ==========================================================
    # AUTO DETECT BUTTON
    # ==========================================================

    def auto_detect(self):

        if self.original is None:
            return

        self.points = self.detect_document(
            self.original
        )

        self.mode = "Original"

        self.redraw()

    # ==========================================================
    # RESET
    # ==========================================================

    def reset(self):

        if self.original is None:
            return

        h, w = self.original.shape[:2]

        margin_x = w * 0.08
        margin_y = h * 0.08

        self.points = np.array(
            [
                [margin_x, margin_y],
                [w - margin_x, margin_y],
                [w - margin_x, h - margin_y],
                [margin_x, h - margin_y]
            ],
            dtype=np.float32
        )

        self.mode = "Original"

        self.redraw()

    # ==========================================================
    # PERSPECTIVE CORRECTION
    # ==========================================================

    def perspective_transform(self):

        points = self.order_points(
            self.points
        )

        tl, tr, br, bl = points

        width1 = np.linalg.norm(
            br - bl
        )

        width2 = np.linalg.norm(
            tr - tl
        )

        height1 = np.linalg.norm(
            tr - br
        )

        height2 = np.linalg.norm(
            tl - bl
        )

        final_width = int(
            max(width1, width2)
        )

        final_height = int(
            max(height1, height2)
        )

        destination = np.array(
            [
                [0, 0],
                [final_width - 1, 0],
                [final_width - 1, final_height - 1],
                [0, final_height - 1]
            ],
            dtype=np.float32
        )

        matrix = cv2.getPerspectiveTransform(
            points,
            destination
        )

        result = cv2.warpPerspective(
            self.original,
            matrix,
            (
                final_width,
                final_height
            ),
            borderMode=cv2.BORDER_REPLICATE
        )

        return result

    # ==========================================================
    # IMAGE ENHANCEMENT
    # ==========================================================

    def enhance(self, image, mode):

        # ---------------------------
        # Original
        # ---------------------------

        if mode == "Original":
            return image

        # ---------------------------
        # Color
        # ---------------------------

        if mode == "Color":

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

            result = cv2.merge(
                [l, a, b]
            )

            return cv2.cvtColor(
                result,
                cv2.COLOR_LAB2BGR
            )

        # ---------------------------
        # Gray
        # ---------------------------

        gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY
        )

        if mode == "Gray":

            return cv2.cvtColor(
                gray,
                cv2.COLOR_GRAY2BGR
            )

        # ---------------------------
        # B&W / Magic Scan
        # ---------------------------

        gray = cv2.GaussianBlur(
            gray,
            (3, 3),
            0
        )

        # Estimate background
        background = cv2.medianBlur(
            gray,
            21
        )

        # Remove uneven lighting
        normalized = cv2.divide(
            gray,
            background,
            scale=255
        )

        if mode == "B&W":

            result = cv2.adaptiveThreshold(
                normalized,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                31,
                11
            )

        else:

            # Magic Scan
            result = cv2.adaptiveThreshold(
                normalized,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                31,
                9
            )

            # Remove small noise
            result = cv2.medianBlur(
                result,
                3
            )

        return cv2.cvtColor(
            result,
            cv2.COLOR_GRAY2BGR
        )

    # ==========================================================
    # CURRENT IMAGE
    # ==========================================================

    def current_image(self):

        if self.original is None:
            return None

        corrected = self.perspective_transform()

        return self.enhance(
            corrected,
            self.mode
        )

    # ==========================================================
    # CHANGE MODE
    # ==========================================================

    def set_mode(self, mode):

        if self.original is None:
            return

        self.mode = mode

        self.redraw()

    # ==========================================================
    # APPLY SCAN
    # ==========================================================

    def apply_scan(self):

        if self.original is None:
            return

        self.mode = "Magic Scan"

        self.status.config(
            text="Scanner enhancement applied."
        )

        self.redraw()

    # ==========================================================
    # REDRAW
    # ==========================================================

    def redraw(self):

        if self.original is None:
            self.canvas.delete("all")
            return

        if self.mode == "Original":

            image = self.original.copy()

            # Draw document outline
            cv2.polylines(
                image,
                [
                    self.points.astype(
                        np.int32
                    )
                ],
                True,
                (255, 190, 0),
                3
            )

        else:

            image = self.current_image()

        rgb = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

        pil = Image.fromarray(rgb)

        canvas_width = max(
            1,
            self.canvas.winfo_width()
        )

        canvas_height = max(
            1,
            self.canvas.winfo_height()
        )

        scale = min(
            canvas_width / pil.width,
            canvas_height / pil.height,
            1.0
        )

        new_width = max(
            1,
            int(pil.width * scale)
        )

        new_height = max(
            1,
            int(pil.height * scale)
        )

        pil = pil.resize(
            (
                new_width,
                new_height
            ),
            Image.Resampling.LANCZOS
        )

        self.display_scale = scale

        self.offset_x = (
            canvas_width - new_width
        ) / 2

        self.offset_y = (
            canvas_height - new_height
        ) / 2

        self.tk_image = ImageTk.PhotoImage(
            pil
        )

        self.canvas.delete("all")

        self.canvas.create_image(
            self.offset_x,
            self.offset_y,
            anchor="nw",
            image=self.tk_image
        )

        # Draw draggable corners
        # only on original mode
        if self.mode == "Original":

            for i, point in enumerate(
                self.points
            ):

                x = (
                    self.offset_x
                    + point[0]
                    * self.display_scale
                )

                y = (
                    self.offset_y
                    + point[1]
                    * self.display_scale
                )

                self.canvas.create_oval(
                    x - 9,
                    y - 9,
                    x + 9,
                    y + 9,
                    fill="#168cff",
                    outline="white",
                    width=2
                )

    # ==========================================================
    # CANVAS COORDINATES
    # ==========================================================

    def canvas_to_image(self, x, y):

        return np.array(
            [
                (x - self.offset_x)
                / self.display_scale,

                (y - self.offset_y)
                / self.display_scale
            ]
        )

    # ==========================================================
    # MOUSE DOWN
    # ==========================================================

    def mouse_down(self, event):

        if (
            self.original is None
            or self.mode != "Original"
        ):
            return

        point = self.canvas_to_image(
            event.x,
            event.y
        )

        distances = np.linalg.norm(
            self.points - point,
            axis=1
        )

        index = int(
            np.argmin(distances)
        )

        if distances[index] < 40:

            self.drag_point = index

    # ==========================================================
    # DRAG CORNER
    # ==========================================================

    def mouse_move(self, event):

        if self.drag_point is None:
            return

        point = self.canvas_to_image(
            event.x,
            event.y
        )

        h, w = self.original.shape[:2]

        x = np.clip(
            point[0],
            0,
            w - 1
        )

        y = np.clip(
            point[1],
            0,
            h - 1
        )

        self.points[
            self.drag_point
        ] = [x, y]

        self.redraw()

    # ==========================================================
    # MOUSE UP
    # ==========================================================

    def mouse_up(self, event):

        self.drag_point = None

    # ==========================================================
    # SAVE JPG
    # ==========================================================

    def save_jpg(self):

        image = self.current_image()

        if image is None:
            return

        filename = filedialog.asksaveasfilename(
            title="Save JPG",
            defaultextension=".jpg",
            filetypes=[
                ("JPEG Image", "*.jpg")
            ]
        )

        if not filename:
            return

        cv2.imwrite(
            filename,
            image,
            [
                cv2.IMWRITE_JPEG_QUALITY,
                95
            ]
        )

        self.status.config(
            text="JPG saved successfully."
        )

    # ==========================================================
    # SAVE PDF
    # ==========================================================

    def save_pdf(self):

        image = self.current_image()

        if image is None:
            return

        filename = filedialog.asksaveasfilename(
            title="Save PDF",
            defaultextension=".pdf",
            filetypes=[
                ("PDF File", "*.pdf")
            ]
        )

        if not filename:
            return

        rgb = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

        pil = Image.fromarray(rgb)

        pil.save(
            filename,
            "PDF",
            resolution=200.0
        )

        self.status.config(
            text="PDF saved successfully."
        )


# ==============================================================
# START PROGRAM
# ==============================================================

if __name__ == '__main__':

    root = tk.Tk()

    app = DocumentScanner(root)

    root.mainloop()
