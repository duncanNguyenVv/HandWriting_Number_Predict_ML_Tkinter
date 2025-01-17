import random
from tkinter import *
from PIL import ImageTk, Image, ImageDraw, ImageGrab
import PIL
import numpy as np
import pickle
import HOG_and_Sobel

def onEnter_btn0(event):
    global img
    img = ImageTk.PhotoImage(Image.open(r'img0_hover.png'))
    b0.config(image=img)

def onEnter_btn1(event):
    global img11
    img11 = ImageTk.PhotoImage(Image.open(r'img1_hover.png'))
    b1.config(image=img11)

def onLeave_btn0(event):
    global img
    img = ImageTk.PhotoImage(Image.open(r'img0.png'))
    b0.config(image=img)

def onLeave_btn1(event):
    global img11
    img11 = ImageTk.PhotoImage(Image.open(r'img1.png'))
    b1.config(image=img11)

def center(image):

    image = image.convert('RGBA')
    image = image.resize((20, 20), PIL.Image.ANTIALIAS)

    raw_image = PIL.Image.new("RGBA", (28, 28), (0, 0, 0))

    x1 = int(.5 * raw_image.size[0]) - int(.5 * image.size[0])
    y1 = int(.5 * raw_image.size[1]) - int(.5 * image.size[1])
    x2 = int(.5 * raw_image.size[0]) + int(.5 * image.size[0])
    y2 = int(.5 * raw_image.size[1]) + int(.5 * image.size[1])

    print(x1,y1,x2,y2)
    print(raw_image.size)
    raw_image.paste(image, box=(x1, y1, x2, y2), mask=image)

    raw_image.convert(raw_image.mode).save("image_for_predict.png")
def predict():
    bbox = image1.getbbox()
    cropped = image1.crop(bbox)
    filename = "image.png"

    cropped.save(filename)

    image = PIL.Image.open("image.png")

    center(image)

    # Step 2. Load image,model,predict
    image_predict = PIL.Image.open('image_for_predict.png').convert("L")
    data = np.asarray(image_predict)
    data = data.flatten()
    print(data.shape)
    ## predict loi thi umcomment dong duoi dien shape vo nhe
    # data = np.reshape(data, (1,784))
    pkl_filename = "pickle_model.pkl"
    with open(pkl_filename, 'rb') as file:
        pickle_model = pickle.load(file)

    x = pickle_model.predict(data)[0]
    # x = random.randint(0,10)
    # Step 3. Print
    deleteText()
    createText(x)

def btn_clicked():
    print("Button Clicked")

def createText(x):
    canvas.create_text(
        709.0, 424.0,
        text=f"THIS IS NUMBER: {x}",
        fill="#d60000",
        font=("Roboto-Bold", int(15.0)),
        tag = "predict_text")

def deleteText():
    canvas.delete("predict_text")
def clear_frame():
   cv.delete('all')
   draw.rectangle((0, 0, 435, 308), fill="black")

def get_x_and_y(event):
    global lasx, lasy
    lasx, lasy = event.x, event.y


def draw_smth(event):
    global lasx, lasy
    cv.create_line((lasx, lasy, event.x, event.y), fill='white', width=2)
    draw.line([lasx, lasy, event.x, event.y], fill='white', width=2)
    lasx, lasy = event.x, event.y

window = Tk()
window.title("Number Predict")
window.geometry("930x449")
window.configure(bg = "#ffffff")
canvas = Canvas(
    window,
    bg = "#ffffff",
    height = 449,
    width = 930,
    bd = 0,
    highlightthickness = 0,
    relief = "ridge")
canvas.place(x = 0, y = 0)

img0 = PhotoImage(file = f"img0.png")
b0 = Button(
    image = img0,
    borderwidth = 0,
    highlightthickness = 0,
    command = clear_frame,
    relief = "flat")

b0.place(
    x = 714, y = 339,
    width = 200,
    height = 50)

img1 = PhotoImage(file = f"img1.png")
b1 = Button(
    image = img1,
    borderwidth = 0,
    highlightthickness = 0,
    command = predict,
    relief = "flat")

b1.place(
    x = 488, y = 339,
    width = 200,
    height = 50)

b0.bind('<Enter>',  onEnter_btn0)
b0.bind('<Leave>',  onLeave_btn0)

b1.bind('<Enter>',  onEnter_btn1)
b1.bind('<Leave>',  onLeave_btn1)

# canvas.create_rectangle(
#     483, 22, 483+435, 22+308,
#     fill = "#000000",
#     outline = "")
###
cv = Canvas(
    window,
    bg = "black",
    height = 308,
    width = 435,
    bd = 0,
    highlightthickness = 0,
    relief = "ridge")
cv.place(x = 483, y = 22)

image1 = PIL.Image.new("RGB", (435, 308), (0, 0, 0))
draw = ImageDraw.Draw(image1)

cv.bind("<Button-1>", get_x_and_y)
cv.bind("<B1-Motion>", draw_smth)
###
# createText('')
# canvas.create_text(
#     709.0, 424.0,
#     text = "THIS IS NUMBER:",
#     fill = "#d60000",
#     font = ("Roboto-Bold", int(15.0)))

background_img = PhotoImage(file = f"background.png")
background = canvas.create_image(
    235.5, 224.5,
    image=background_img)

window.resizable(False, False)
window.mainloop()
