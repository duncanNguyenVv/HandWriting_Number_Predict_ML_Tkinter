from PIL import ImageTk, Image, ImageDraw, ImageGrab
import PIL
from tkinter import *
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
from skimage.io import imread
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

def extract_features(X):
    F = np.empty((len(X), 2))
    F[:, 0] = X.mean(axis=1)
    for r in range(len(X)):
        img = X[r].reshape(16, 16)
        F[r, 1] = -(np.mean(np.abs(img - np.flip(img, axis=0))) + np.mean(np.abs(img - np.flip(img, axis=1)))) / 2
    return F

width = 200
height = 200
center = height//2
white = (255, 255, 255)
green = (0,128,0)
black = (0, 0, 0)

def test():
    data = pd.read_csv('ttt.csv')
    return data.iloc[[0]]

def save():
    bbox = image1.getbbox()
    # print(bbox,bbox[3])
    cropped = image1.crop(bbox)
    filename = "image.png"

    cropped.save(filename)

    image = PIL.Image.open("image.png")
    image = image.resize((16, 16), PIL.Image.ANTIALIAS)
    # image = PIL.ImageTk.PhotoImage(image)
    image.save('image1.png')

    im = imread("image1.png", 0)
    image_predict = PIL.Image.open('image1.png').convert("L")
    pix_val = np.array(image_predict.getdata())
    data = np.asarray(image_predict)
    # plt.imshow(data, cmap=plt.cm.binary)
    data = data.flatten()
    data = np.reshape(data, (1,256))

    # x = [0]
    data = test()
    data = np.array(data)
    data = data[0,1:]
    data = np.reshape(data, (1, 256))
    x = pickle_model.predict(data)
    # summarize some details about the image
    print(data.shape)
    var.set(f"This is number: {x[0]}")
    label.pack()

def get_x_and_y(event):
    global lasx, lasy
    lasx, lasy = event.x, event.y


def draw_smth(event):
    global lasx, lasy
    cv.create_line((lasx, lasy, event.x, event.y), fill='white', width=2)
    draw.line([lasx, lasy, event.x, event.y], fill='white', width=2)
    lasx, lasy = event.x, event.y

root = Tk()
root.title("Number Predict")
root.resizable(width=False, height=False)
# Tkinter create a canvas to draw on
cv = Canvas(root, width=width, height=height, bg='black')
cv.pack()

var = StringVar()
label = Label(root, textvariable=var, relief=RAISED)

# PIL create an empty image and draw object to draw on
# memory only, not visible
image1 = PIL.Image.new("RGB", (width, height), black)
draw = ImageDraw.Draw(image1)



cv.pack(anchor='nw', fill='both', expand=1)


cv.bind("<Button-1>", get_x_and_y)
cv.bind("<B1-Motion>", draw_smth)

button=Button(text="Predict",command=save)
button.pack(pady=10)

def clear_frame():
   cv.delete('all')
   draw.rectangle((0, 0, width, height), fill="black")
Button(text="Clear", command=clear_frame).place(x=150,y=213.6)

pkl_filename = "pickle_model.pkl"
with open(pkl_filename, 'rb') as file:
    pickle_model = pickle.load(file)

root.mainloop()
