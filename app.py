import streamlit as st
import time

import os
import pickle as pkl
from utils.data import read_pickle
from run_inference import run_inference
from visualize_heatmap import visualize_heatmap

# st.image('images/sim.png', width=200)
st.set_page_config(page_title="DPnSSM", page_icon="♓", layout="centered")
st.title("Clustering Time Series with Nonlinear Dynamics ♓")

with st.form("form"):
    uploaded_files = st.file_uploader("Upload your spike data in pickle format with extension .p", accept_multiple_files=True)

    iter = st.slider('Number of iterations', 1, 100000, 1)

    st.info('Larger numbers would increase accuracy but would significantly increase processing time', icon="ℹ️")
    
    submitted = st.form_submit_button("Run")
    
    if submitted:
        # supress to 2 files max
        if len(uploaded_files) > 2 or len(uploaded_files) < 1:
            st.error("Please upload at most 2 files only")
            st.stop()
        else:
            st.success('Submitted', icon='✅')
    

for i, uploaded_file in enumerate(uploaded_files):
    st.write(uploaded_file.name)
    if uploaded_file is not None:
        bytes_data = uploaded_file.getvalue()
            
        if not os.path.exists('uploads'):
            os.makedirs('uploads')
                
        with open(os.path.join('uploads', uploaded_file.name), 'wb') as f:
            f.write(bytes_data)
                
        file = read_pickle(os.path.join('uploads', uploaded_file.name)) 
        
        key = uploaded_file.name

        if st.button("Cancel process", key=key):
            st.warning('Process cancelled', icon='⚠️')
            st.stop()
            
        st.toast('Running inference', icon='⏳')     
        
        run_inference(os.path.join('uploads', uploaded_file.name), iter)
            
        st.success('Inference Completed!', icon='✅')
        
        if len(uploaded_files) == 1:
            visualize_heatmap('inference/pickle.p')
            
            st.write(uploaded_files[0].name)
            st.image('images/heatmap.png')
            os.remove('images/heatmap.png')
        
        if len(uploaded_files) == 2 and i == 1:
            visualize_heatmap('inference/pickle.p')
            visualize_heatmap('inference/pickle_2.p')
            
            col1, col2 = st.columns(2)
            
            col1.write(uploaded_files[0].name)
            col1.image('images/heatmap.png')
            os.remove('images/heatmap.png')
            
            col2.write(uploaded_files[1].name)
            col2.image('images/heatmap_2.png')
            os.remove('images/heatmap_2.png')
        
    else: 
        pass
        

        

    
