import cv2
import numpy as np
import customtkinter as ctk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk

ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

class CamScannerDesktop(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("CamScanner Desktop")
        self.geometry("1100x800")

        self.orig_image = None
        self.processed_image = None
        self.pts_orig = []
        self.pts_display = []
        self.selected_point = None
        self.scale_factor = 1.0

        self.setup_ui()

    def setup_ui(self):
        # Top Control Bar
        self.btn_frame = ctk.CTkFrame(self)
        self.btn_frame.pack(side="top", fill="x", padx=15, pady=10)

        self.btn_load = ctk.CTkButton(self.btn_frame, text="📁 فتح صورة", command=self.load_image, font=("Arial", 13, "bold"))
        self.btn_load.pack(side="left", padx=8, pady=8)

        self.btn_reset = ctk.CTkButton(self.btn_frame, text="🔄 إعادة ضبط الزوايا", command=self.auto_detect_corners, state="disabled", fg_color="#555555")
        self.btn_reset.pack(side="left", padx=8, pady=8)

        # Filter Options
        self.lbl_filter = ctk.CTkLabel(self.btn_frame, text="الفلتر:", font=("Arial", 12, "bold"))
        self.lbl_filter.pack(side="left", padx=(15, 5))

        self.filter_var = ctk.StringVar(value="Magic Color")
        self.filter_menu = ctk.CTkOptionMenu(
            self.btn_frame, 
            values=["Magic Color", "B&W Scanner", "Original"],
            variable=self.filter_var
        )
        self.filter_menu.pack(side="left", padx=5)

        self.btn_crop = ctk.CTkButton(self.btn_frame, text="✂️ قص وتطبيق الفلتر", command=self.crop_and_enhance, state="disabled", fg_color="#1f538d", font=("Arial", 13, "bold"))
        self.btn_crop.pack(side="left", padx=15, pady=8)

        self.btn_save = ctk.CTkButton(self.btn_frame, text="💾 حفظ المستند", command=self.save_image, state="disabled", fg_color="green", font=("Arial", 13, "bold"))
        self.btn_save.pack(side="right", padx=8, pady=8)

        # Canvas Area
        self.canvas = ctk.CTkCanvas(self, bg="#1e1e1e", highlightthickness=0)
        self.canvas.pack(expand=True, fill="both", padx=15, pady=10)

        # Mouse Events for Point Dragging
        self.canvas.bind("<ButtonPress-1>", self.on_mouse_down)
        self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_mouse_up)

    def load_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp")])
        if not file_path:
            return

        self.orig_image = cv2.imread(file_path)
        self.auto_detect_corners()
        self.btn_reset.configure(state="normal")
        self.btn_crop.configure(state="normal")
        self.btn_save.configure(state="disabled")

    def auto_detect_corners(self):
        if self.orig_image is None:
            return

        h, w = self.orig_image.shape[:2]
        gray = cv2.cvtColor(self.orig_image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edged = cv2.Canny(blurred, 75, 200)

        contours, _ = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

        doc_cnt = None
        for c in contours:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            if len(approx) == 4:
                doc_cnt = approx
                break

        if doc_cnt is not None:
            pts = doc_cnt.reshape(4, 2).astype("float32")
            self.pts_orig = self.order_points(pts)
        else:
            margin_w, margin_h = w * 0.1, h * 0.1
            self.pts_orig = np.array([
                [margin_w, margin_h],
                [w - margin_w, margin_h],
                [w - margin_w, h - margin_h],
                [margin_w, h - margin_h]
            ], dtype="float32")

        self.redraw_canvas()

    def redraw_canvas(self):
        if self.orig_image is None:
            return

        self.canvas.delete("all")
        canvas_w = self.canvas.winfo_width() or 800
        canvas_h = self.canvas.winfo_height() or 550

        img_h, img_w = self.orig_image.shape[:2]
        self.scale_factor = min(canvas_w / img_w, canvas_h / img_h)
        new_w, new_h = int(img_w * self.scale_factor), int(img_h * self.scale_factor)

        rgb_img = cv2.cvtColor(self.orig_image, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_img).resize((new_w, new_h), Image.Resampling.LANCZOS)
        self.tk_image = ImageTk.PhotoImage(pil_img)

        self.offset_x = (canvas_w - new_w) // 2
        self.offset_y = (canvas_h - new_h) // 2
        self.canvas.create_image(self.offset_x, self.offset_y, anchor="nw", image=self.tk_image)

        self.pts_display = []
        for pt in self.pts_orig:
            x = pt[0] * self.scale_factor + self.offset_x
            y = pt[1] * self.scale_factor + self.offset_y
            self.pts_display.append([x, y])

        for i in range(4):
            pt1 = self.pts_display[i]
            pt2 = self.pts_display[(i + 1) % 4]
            self.canvas.create_line(pt1[0], pt1[1], pt2[0], pt2[1], fill="#00FF00", width=2)

        for i, (x, y) in enumerate(self.pts_display):
            self.canvas.create_oval(x - 8, y - 8, x + 8, y + 8, fill="#FF0000", outline="#FFFFFF", width=2)

    def on_mouse_down(self, event):
        for i, (x, y) in enumerate(self.pts_display):
            if np.hypot(event.x - x, event.y - y) < 15:
                self.selected_point = i
                break

    def on_mouse_drag(self, event):
        if self.selected_point is not None:
            x, y = event.x, event.y
            self.pts_display[self.selected_point] = [x, y]

            orig_x = (x - self.offset_x) / self.scale_factor
            orig_y = (y - self.offset_y) / self.scale_factor

            img_h, img_w = self.orig_image.shape[:2]
            orig_x = np.clip(orig_x, 0, img_w)
            orig_y = np.clip(orig_y, 0, img_h)

            self.pts_orig[self.selected_point] = [orig_x, orig_y]
            self.redraw_canvas()

    def on_mouse_up(self, event):
        self.selected_point = None

    def order_points(self, pts):
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        return rect

    def crop_and_enhance(self):
        if self.orig_image is None or len(self.pts_orig) != 4:
            return

        rect = self.pts_orig
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
        warped = cv2.warpPerspective(self.orig_image, M, (maxWidth, maxHeight))

        filter_type = self.filter_var.get()

        if filter_type == "B&W Scanner":
            gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
            self.processed_image = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 10
            )
        elif filter_type == "Magic Color":
            # Color enhancement filter
            norm = cv2.normalize(warped, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
            self.processed_image = cv2.detailEnhance(norm, sigma_s=10, sigma_r=0.15)
        else: # Original
            self.processed_image = warped

        # Display result
        self.canvas.delete("all")
        canvas_w = self.canvas.winfo_width() or 800
        canvas_h = self.canvas.winfo_height() or 550

        scale = min(canvas_w / maxWidth, canvas_h / maxHeight)
        new_w, new_h = int(maxWidth * scale), int(maxHeight * scale)

        if len(self.processed_image.shape) == 2:
            rgb_proc = cv2.cvtColor(self.processed_image, cv2.COLOR_GRAY2RGB)
        else:
            rgb_proc = cv2.cvtColor(self.processed_image, cv2.COLOR_BGR2RGB)

        pil_img = Image.fromarray(rgb_proc).resize((new_w, new_h), Image.Resampling.LANCZOS)
        self.tk_image = ImageTk.PhotoImage(pil_img)

        offset_x = (canvas_w - new_w) // 2
        offset_y = (canvas_h - new_h) // 2
        self.canvas.create_image(offset_x, offset_y, anchor="nw", image=self.tk_image)

        self.btn_save.configure(state="normal")

    def save_image(self):
        if self.processed_image is None:
            return
        save_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG file", "*.png"), ("JPEG file", "*.jpg")])
        if save_path:
            cv2.imwrite(save_path, self.processed_image)
            messagebox.showinfo("تم الحفظ", "تم حفظ المستند بنجاح!")

if __name__ == "__main__":
    app = CamScannerDesktop()
    app.mainloop()
