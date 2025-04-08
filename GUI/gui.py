import os
import tkinter as tk
from tkinter import Canvas, Button, filedialog, ttk

import keras
from PIL import Image, ImageTk
from gui_main import make_prediction, load_trained_model, make_gradcam_heatmap, \
    get_img_array, apply_gradcam_overlay

window = tk.Tk()
window.geometry("1000x550")
window.configure(bg="#FFFFFF")
window.resizable(False, False)

selected_model = tk.StringVar()
selected_model.set("Válassz modellt")

selected_conv = tk.StringVar()
selected_conv.set("Válassz réteget")

canvas = Canvas(
    window,
    bg="#FFFFFF",
    height=550,
    width=1000,
    bd=0,
    highlightthickness=0,
    relief="ridge"
)
canvas.place(x=0, y=0)

def create_rectangle(canvas, x1, y1, x2, y2, radius=25, **kwargs):
    points = [
        x1+radius, y1, x2-radius, y1, x2, y1, x2, y1+radius,
        x2, y2-radius, x2, y2, x2-radius, y2, x1+radius, y2,
        x1, y2, x1, y2-radius, x1, y1+radius, x1, y1
    ]
    return canvas.create_polygon(points, **kwargs, smooth=True)

canvas.create_rectangle(0, 0, 1000, 550, fill="#FF8C00", outline="")
create_rectangle(canvas, 377, 35, 977, 515, radius=25, fill="#FF8C00", outline="Brown")

def create_button(canvas, x1, y1, x2, y2, text, command, radius=30):
    create_rectangle(canvas, x1, y1, x2, y2, radius=radius, fill="#A52A2A", outline="")
    btn = Button(
        window, text=text, font=("ComicSansMS", 12), fg="white", bg="#A52A2A", activebackground="#A52A2A",
        relief="flat", command=command, width=17, height=1, borderwidth=0
    )
    btn.place(x=x1+10, y=y1+5)
    return btn

selected_file_path = ""
def select_and_display_image():
    global selected_file_path
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg;*.jpeg;*.png;*.bmp")])
    if file_path:
        selected_file_path = file_path
        img = Image.open(file_path)
        img = img.resize((580, 460))
        img_tk = ImageTk.PhotoImage(img)
        canvas.create_image(677, 275, anchor="center", image=img_tk)
        canvas.image = img_tk

def create_label(canvas, x1, y1, x2, y2, text, radius=15):
    create_rectangle(canvas,x1, y1, x2, y2, radius, fill="#A52A2A", outline="", width=0)
    canvas.create_text((x1 + x2) / 2, (y1 + y2) / 2, text=text, font=("Comic Sans MS", 10), fill="white")


def inference(selected_model, selected_file_path, canvas):
    model_name = selected_model.get()

    if model_name == "Válassz modellt":
        create_label(canvas, 50, 350, 225, 390, "Nem választottál modellt!")
        return
    if not selected_file_path:
        create_label(canvas, 50, 350, 225, 390, "Nem választottál képet!")
        return

    loaded_model = load_trained_model(os.path.join('../checkpoints', model_name))
    confidence, predicted_class = make_prediction(loaded_model, selected_file_path)

    classes = ["macska", "kutya", "ember"]
    predicted_label = classes[predicted_class] if predicted_class < len(classes) else "Nem felismerhető"

    prediction_text = f"A képen látható: {predicted_label}\nEsély: {confidence:.2f}%"
    create_label(canvas, 50, 350, 225, 390, prediction_text)


def visualize(selected_model, selected_conv, canvas):
    model_name = selected_model.get()
    last_conv_layer_name = selected_conv.get()

    if not selected_file_path:
        create_label(canvas, 50, 350, 225, 390, "Nem választottál képet!")
        return
    if model_name == "Válassz modellt":
        create_label(canvas, 50, 350, 225, 390, "Nem választottál modellt!")
        return
    if last_conv_layer_name == "Válassz réteget":
        create_label(canvas, 50, 350, 225, 390, "Nem választottál réteget!")
        return

    loaded_model = load_trained_model(os.path.join('../checkpoints', model_name))

    img_size = (256, 256)
    img_array = keras.applications.xception.preprocess_input(get_img_array(selected_file_path, size=img_size))

    heatmap = make_gradcam_heatmap(img_array, loaded_model, last_conv_layer_name)

    heatmap_img = apply_gradcam_overlay(selected_file_path, heatmap)

    heatmap_img = heatmap_img.resize((580, 460))
    heatmap_img_tk = ImageTk.PhotoImage(heatmap_img)

    canvas.create_image(677, 275, anchor="center", image=heatmap_img_tk)
    canvas.image = heatmap_img_tk

def create_model_combobox():
    model_directory = "../checkpoints"
    model_options = [f for f in os.listdir(model_directory) if f.endswith(".keras")]
    model_dropdown = ttk.Combobox(window, textvariable=selected_model, values=model_options, font=("ComicSansMS", 8),
                                  state="readonly")
    model_dropdown.place(x=50, y=140, width=175, height=30)
    window.option_add('*TCombobox*Listbox.background', '#A52A2A')
    window.option_add('*TCombobox*Listbox.foreground', 'white')
    window.option_add('*TCombobox*Foreground', 'white')
    window.option_add('*TCombobox*Background', '#A52A2A')
    window.option_add('*TCombobox*Listbox.selectBackground', 'orange')
    window.option_add('*TCombobox*Listbox.selectForeground', 'black')

def create_layer_combobox():
    conv_options = ["conv1","conv2d","conv2d_1","conv2d_2","conv2d_3","conv2d_4","conv2d_5","conv2d_6","conv2d_7"
                     ,"conv2d_8","conv2d_9","conv2d_10","conv2d_11"]
    conv_dropdown = ttk.Combobox(window, textvariable=selected_conv, values=conv_options, font=("ComicSansMS", 8),
                                  state="readonly")
    conv_dropdown.place(x=235, y=285, width=125, height=30)
    window.option_add('*TCombobox*Listbox.background', '#A52A2A')
    window.option_add('*TCombobox*Listbox.foreground', 'white')
    window.option_add('*TCombobox*Foreground', 'white')
    window.option_add('*TCombobox*Background', '#A52A2A')
    window.option_add('*TCombobox*Listbox.selectBackground', 'orange')
    window.option_add('*TCombobox*Listbox.selectForeground', 'black')


create_button(canvas, 50, 58, 225, 98, "Kép betöltése",lambda: select_and_display_image())
create_model_combobox()
create_button(canvas, 50, 205, 225, 245, "Futtatás", lambda: inference(selected_model, selected_file_path, canvas))
create_button(canvas, 50, 280, 225, 320, "Vizualizálás", lambda: visualize(selected_model,selected_conv,canvas))
create_layer_combobox()

window.mainloop()