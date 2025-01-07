import streamlit as st

st.markdown("## Check out the code here !!!")
with open('model/weights.h5', 'rb') as f:
    weights_data = f.read()

# Provide a download button for the user
st.download_button(
    label='Download Weights',
    data=weights_data,  # Pass the binary content of the file
    file_name='weights.h5',  # Specify the name of the downloaded file
    mime='application/octet-stream'  # MIME type for binary files
)

st.markdown("## Download the pretrained model from here !!!")
with open('notebook/U_Net_Sratch (1).ipynb', 'rb') as f:
    weights_data = f.read()

# Provide a download button for the user
st.download_button(
    label='Python Notebook',
    data=weights_data,  # Pass the binary content of the file
    file_name='U_Net_Sratch (1).ipynb',  # Specify the name of the downloaded file
    mime='application/octet-stream'  # MIME type for binary files
)
