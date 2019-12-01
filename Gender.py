from fastai import *
from fastai.vision import *

import tkinter as tk
from tkinter import filedialog

classes=["Male","Female"]

root = tk.Tk()
root.withdraw()
file_path = filedialog.askopenfilename()
# print(file_path)

img = open_image(file_path)

path=Path(r"C:\Users\Asus\Desktop\Minor") #path of export.pkl
learner = load_learner(path)
pred_class,pred_idx,outputs = learner.predict(img)

print(pred_class)