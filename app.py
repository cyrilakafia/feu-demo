import streamlit as st
import os
import torch
from inf import run_inference
from visualize_heatmap import viz_heatmap

# Set page title and favicon.
st.set_page_config(page_title="DPnSSM", page_icon="♓", layout="wide")

# Set up the title and logo
col1, mid1, col2, mid2, col3 = st.columns([1,1,1,1,20])
with col1:
    st.image('./images/AZALab_Logo_Social Profile Image_Mark Only.jpg', width=100)

with col2:
    st.image('./images/sparsitas.jpg', width=75)

with col3:
    st.title("FEU - Functional Encoding Units")


# Set up a test case
if 'test' not in st.session_state:
    st.session_state['test'] = False

# Create two columns for the split view
left_column, right_column = st.columns([2, 3])

with left_column:
    with st.form("form"):
        uploaded_file = st.file_uploader("Upload your spike data in pickle format with extension .p", accept_multiple_files=False)
        iter = st.slider('Number of iterations', 1, 1500, 1)
        st.info('Larger numbers would increase accuracy but would significantly increase processing time', icon="ℹ️")
        submitted = st.form_submit_button("Run")

    # Test run button
    if st.button("Run test with default data", key="test_button"):
        st.session_state['test'] = True
        st.write("Running test...")
        iter = 1  
        submitted = True  

    if submitted:
        if st.session_state['test']:
            # Handle test run
            st.toast('Running test inference', icon='⏳')
            output = run_inference('default/sim1231_true.p', title='1', device='cpu', iterations=iter, seed=None)
            st.toast('Test Inference Completed!', icon='✅')

        elif uploaded_file:
            if uploaded_file is not None:
                bytes_data = uploaded_file.getvalue()

                if not os.path.exists('uploads'):
                    os.makedirs('uploads')

                with open(os.path.join('uploads', uploaded_file.name), 'wb') as f:
                    f.write(bytes_data)

                key = uploaded_file.name
                if st.button("❌ Cancel", key=key):
                    st.warning('Process cancelled', icon='⚠️')
                    st.stop()

                st.toast('Running inference', icon='⏳')     
                output = run_inference(os.path.join('uploads', uploaded_file.name), title='1', device='cpu', iterations=iter, seed=None)
                st.toast('Inference Completed!', icon='✅')

        else: 
            st.warning('No file uploaded', icon='⚠️')
            st.stop()

        with right_column:
            if st.session_state['test'] and submitted:
                best_assigns = viz_heatmap('1', iter = iter)
                st.image('images/heatmap.png')
                os.remove('images/heatmap.png')

            elif uploaded_file:
                best_assigns = viz_heatmap('1', iter = iter)
                st.image('images/heatmap.png')
                os.remove('images/heatmap.png')

                # Remove uploaded file
                os.remove(os.path.join('uploads', uploaded_file.name))

            
        @st.cache_data
        def convert_df(df):
        # IMPORTANT: Cache the conversion to prevent computation on every rerun
            return df.to_csv().encode('utf-8')
        
        csv = convert_df(best_assigns)

        st.write(best_assigns)
        
        col1, col2 = st.columns(2)

        with col1:
            st.download_button(
                label="🗂️ Download cluster assignments",
                data= csv,
                file_name=f'best_assigns.csv',
                mime="text/csv",
            )

        with col2:
            st.download_button(
                label="🗂️ Download cluster parameters",
                data="outputs/sim1_params.tsv",
                file_name="sim1_params.tsv",
                mime="text/tab-separated-values",
            )
    if st.session_state['test'] and submitted:
        st.session_state['test'] = False
