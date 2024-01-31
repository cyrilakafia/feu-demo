import streamlit as st
import os
import torch
from inf import run_inference
from visualize_heatmap import visualize_heatmap

st.set_page_config(page_title="DPnSSM", page_icon="♓", layout="wide")

col1, mid, col2 = st.columns([1,1,20])
with col1:
    st.image('./images/AZALab_Logo_Social Profile Image_Mark Only.jpg', width=100)
with col2:
    st.title("FEU - Functional Encoding Units")

# Create two columns for the split view
left_column, right_column = st.columns([2, 3])

with left_column:
    with st.form("form"):
        uploaded_files = st.file_uploader("Upload your spike data in pickle format with extension .p", accept_multiple_files=True)
        iter = st.slider('Number of iterations', 1, 1500, 1)
        st.info('Larger numbers would increase accuracy but would significantly increase processing time', icon="ℹ️")
        submitted = st.form_submit_button("Run")

    if submitted:
        for i, uploaded_file in enumerate(uploaded_files):
            if uploaded_file is not None:
                bytes_data = uploaded_file.getvalue()

                if not os.path.exists('uploads'):
                    os.makedirs('uploads')

                with open(os.path.join('uploads', uploaded_file.name), 'wb') as f:
                    f.write(bytes_data)

                key = uploaded_file.name
                if st.button("❌ Cancel process", key=key):
                    st.warning('Process cancelled', icon='⚠️')
                    st.stop()

                st.toast('Running inference', icon='⏳')     
                output = run_inference(os.path.join('uploads', uploaded_file.name), title='1', device='cpu', iterations=iter, seed=None)
                st.toast('Inference Completed!', icon='✅')

            with right_column:
                if len(uploaded_files) == 1:
                    visualize_heatmap('inference/pickle.p')
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

            os.remove(os.path.join('uploads', uploaded_file.name))

            # Download buttons are now here, below the "Cancel process" button
            st.download_button(
                label="🗂️ Download cluster assignments",
                data=f'outputs/sim1_assigns.csv',
                file_name=f'sim1_assigns.csv',
                mime="text/csv",
            )

            st.download_button(
                label="🗂️ Download cluster parameters",
                data="outputs/sim1_params.tsv",
                file_name="sim1_params.tsv",
                mime="text/tab-separated-values",
            )
