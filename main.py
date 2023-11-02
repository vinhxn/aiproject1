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
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    transform_norm = transforms.Compose([transforms.ToTensor(),
    transforms.Resize((512,512)),transforms.Normalize(mean, std)])
    # get normalized image
    img_normalized = transform_norm(image).float()
    img_normalized = img_normalized.unsqueeze_(0)
    img_normalized = img_normalized.to(device)
    # print(img_normalized.shape)
    with torch.no_grad():
        #model.eval()
        output =model(img_normalized)
        index = output.data.cpu().numpy().argmax()
        print(index)
        class_name = classes[index]
        return class_name

#def load_model_from_google_drive(cloud_model_location, model_path):
#    save_dest = Path('model')
#    save_dest.mkdir(exist_ok=True)
#    
#    #f_model_path = Path("model/skyAR_coord_resnet50.pt")
#    f_model_path = Path(model_path)
#    if not f_model_path.exists():
#        with st.spinner("Downloading model... this may take awhile! \n Don't stop it!"):
#            gdd(cloud_model_location, f_model_path)    
#    model = torch.load(model_path, map_location=device)
#    # Disbribute the model to all GPU's
#    model = model.module
#    # set model to run on GPU or CPU absed on availibility
#    model.to(device)
#    model.eval()
#    return model

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
#cloud_model_location='1ClxtIfqn1qZH1C1tm5QGyinHncx0bCoU'

# load model
model = load_model(model_path)
#model = load_model_from_google_drive(cloud_model_location, model_path)

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
