import customtkinter as ctk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
from ultralytics import YOLO
import os
from PIL import Image, ImageDraw, ImageFont


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
current_dir = os.getcwd()


def draw_vietnamese_text(image, text, position, color):
    image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(image_pil)
    font_path = "./fonts/Roboto-Regular.ttf"
    font_size = 18
    font = ImageFont.truetype(font_path, font_size)
    draw.text(position, text, font=font, fill=color)
    return cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)


def draw_box_with_label(img, box, label):
    
    x_min, y_min, x_max, y_max = map(int, box)
    box_height = y_max - y_min
    box_width = x_max - x_min
    label_x = x_min
    label_y = y_min - 10
    if label_y - 20 < 0:
        label_y = y_min + 10

    POSITION_PER_LABEL = {
        "An toàn": (label_x + 80, label_y + 10),
        "Không an toàn": (label_x + 150, label_y + 10),
        "Mũ": (label_x + 50, label_y + 10),
        "Áo": (label_x + 50, label_y + 10)
    }


    COLOR_PER_LABEL = {
        "An toàn": (56, 179, 92),
        "Không an toàn": (65, 65, 251),
        "Mũ": (69, 193, 255),
        "Áo": (119, 80, 46)
    }

    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), COLOR_PER_LABEL[label], 2)
    cv2.rectangle(img, (label_x, label_y - 10), POSITION_PER_LABEL[label], COLOR_PER_LABEL[label], -1)
    img = draw_vietnamese_text(img, label, (label_x + 5, label_y - 9), (255, 255, 255))
    return img

def is_inside(box_a, box_b):
    x_min_a, y_min_a, x_max_a, y_max_a = box_a
    x_min_b, y_min_b, x_max_b, y_max_b = box_b

    center_a_x = (x_min_a + x_max_a) / 2
    center_a_y = (y_min_a + y_max_a) / 2

    return x_min_b <= center_a_x <= x_max_b and y_min_b <= center_a_y <= y_max_b


def refresh_image():
    if not hasattr(refresh_image, "image_path") or not refresh_image.image_path:
        return
    process_image(refresh_image.image_path)


def process_image(image_path=None):
    global img_result, img_display, total_people, people_without_full_clothing

    if image_path is None:
        image_path = filedialog.askopenfilename(title="Chọn ảnh", filetypes=[("Image files", "*.jpg *.jpeg *.png")])
        if not image_path:
            return
        refresh_image.image_path = image_path

    loading_label.configure(image=loading_gif)
    loading_label.image = loading_gif
    app.update()

    image_label.configure(text="")
    app.update()

    yolov8_model = YOLO(f"./models/last.pt")
    results = yolov8_model([image_path])
    img = cv2.imread(image_path)

    for result in results:
        boxes = result.boxes
        xyxy = boxes.xyxy.cpu()
        conf = boxes.conf.cpu()
        cls = boxes.cls.cpu()

        masks = {
            0: (cls == 0).nonzero(as_tuple=True)[0],
            1: (cls == 1).nonzero(as_tuple=True)[0],
            2: (cls == 2).nonzero(as_tuple=True)[0]
        }

        total_people = 0
        people_without_full_clothing = 0
        for person_idx in masks[2]:
            # Tăng confidence score cho người để nhận diện chính xác hơn nè (0.5 -> 0.7): 
            if conf[person_idx] < 0.7:
                continue

            total_people += 1
            person_box = xyxy[person_idx]
            contains_hat = any(is_inside(xyxy[hat_idx], person_box) for hat_idx in masks[1])
            contains_shirt = any(is_inside(xyxy[shirt_idx], person_box) for shirt_idx in masks[0])
            label = "An toàn" if contains_hat and contains_shirt else "Không an toàn"
            if not (contains_hat and contains_shirt):
                people_without_full_clothing += 1

            img = draw_box_with_label(img, person_box, label)

        if show_boxes_var.get():
            # Hiển thị label cho mũ
            for hat_idx in masks[1]:
                hat_box = xyxy[hat_idx]
                img = draw_box_with_label(img, hat_box, "Mũ")

            # Hiển thị label cho mũ
            for shirt_idx in masks[0]:
                shirt_box = xyxy[shirt_idx]
                img = draw_box_with_label(img, shirt_box, "Áo")

    img_result = img
    img_display = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))

    # Loading Animation
    image_label.configure(image=img_display)
    image_label.image = img_display
    loading_label.configure(image="")

    result_label.configure(text=f"Tổng số người: {total_people}\nSố người không đủ mũ và áo: {people_without_full_clothing}")

# Hàm lưu ảnh
def save_image():
    try: 
        if img_result is None:
            return
    except:
        messagebox.showerror("Error", "Không có ảnh để lưu.")
        return
    save_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG files", "*.png")])
    if save_path:
        try:
            cv2.imwrite(save_path, img_result)
            messagebox.showinfo("Success", f"Đã lưu ảnh tại: {save_path}")
        except Exception as e:
            messagebox.showerror("Error", f"Không thể lưu ảnh: {e}")

# Khởi tạo giao diện
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

app = ctk.CTk()
app.title("YOLOv8 Detection App")
app.geometry("1200x800")

# Frame chính
main_frame = ctk.CTkFrame(app)
main_frame.pack(fill="both", expand=True, padx=20, pady=20)

# Bố cục
left_frame = ctk.CTkFrame(main_frame, width=600, corner_radius=10)
right_frame = ctk.CTkFrame(main_frame, width=250, corner_radius=10)

left_frame.pack(side="left", fill="both", expand=True, padx=10)
right_frame.pack(side="right", fill="y", padx=10)

# Hiển thị ảnh
image_label = ctk.CTkLabel(left_frame, text="Chọn một ảnh để hiển thị kết quả", font=("Arial", 16), fg_color="gray")
image_label.pack(fill="both", expand=True, padx=10, pady=10)

# Hiển thị trạng thái xử lý
loading_gif = Image.open("./images/loading.gif")
loading_gif = loading_gif.resize((50, 50))  # Thay đổi kích thước GIF
loading_gif = ImageTk.PhotoImage(loading_gif)  # Chuyển đổi GIF sang ImageTk
loading_label = ctk.CTkLabel(left_frame, text="")
loading_label.pack(pady=20)

# Nút chọn và lưu
btn_select = ctk.CTkButton(right_frame, text="Chọn Ảnh", command=lambda: process_image(None), font=("Arial", 16), width=200)
btn_select.pack(pady=20)

btn_save = ctk.CTkButton(right_frame, text="Lưu Kết Quả", command=save_image, font=("Arial", 16), width=200)
btn_save.pack(pady=20)

# Nút tick hiển thị box
show_boxes_var = ctk.BooleanVar(value=False)
checkbox_show_boxes = ctk.CTkCheckBox(right_frame, text="Hiển thị box", variable=show_boxes_var, font=("Arial", 16))
checkbox_show_boxes.pack(pady=20)

# Liên kết nút tick với làm mới ảnh
show_boxes_var.trace_add("write", lambda *args: refresh_image())

# Hiển thị kết quả
result_label = ctk.CTkLabel(right_frame, text="", font=("Arial", 14))
result_label.pack(pady=20)

app.mainloop()