import streamlit as st
import numpy as np
import pandas as pd
from torchvision import models, transforms
import torch
from skimage import io, transform

# UI
st.title("AI Project")
st.title("Ocular Diease Recognition")


# class names
classes = { 0: "Normal",
            1: "Diabetes",
            2: "Glaucoma",
            3: "Cataract",
            4: "Age related Macular Degeneration",
            5: "Hypertension",
            6: "Pathological Myopia",
            7: "Other diseases/abnormalities"
          }

# check if CUDA is available
device_name = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(device_name)

def predict_image(image_path, model):
    image = io.imread(image_path)
    transform_img =transforms.Compose([transforms.ToTensor(), transforms.Resize((512,512))])
    image = transform_img(image)
    X = None
    with torch.no_grad():        
        X = image.to(device, dtype=torch.float)
        # forward pass image
        y_val = model(X.view(-1, 3, 512, 512))

        # get argmax of predicted tensor, which is our label
        predicted = y_val.data.cpu().numpy().argmax()
        #print("Class index: {}".format(predicted))
        class_name = classes[predicted]
        return class_name
   
def load_model(model_path):    
    # load model
    model = torch.load(model_path, map_location=device)
    # Disbribute the model to all GPU's
    model = model.module
    # set model to run on GPU or CPU absed on availibility
    model.to(device)
    model.eval()
    return model

# model path
model_path = 'model/bt_resnet50_v2.pth'
# model from Google Drive

# load model
model = load_model(model_path)

# display image
uploaded_file = st.file_uploader("Choose a jpg file", type=['jpg'])
if uploaded_file is not None:
    # To read file as bytes:
    #st.write("Filename: ", uploaded_file.name)
    #image = Image.open(uploaded_file).convert('RGB')
    image = io.imread(uploaded_file)
    st.image(image, use_column_width=True)

clicked = st.button('Predict')
if clicked:
    if uploaded_file is None:
        st.write("Please upload a JPG file!")
    else:
        # classify image
        predict_class = predict_image(uploaded_file, model)
        # write classification
        st.write("Predicted classification:")
        st.header(predict_class)
