import streamlit as st
import torch
from pyannote.audio import Pipeline
from pyannote.audio.pipelines.utils.hook import ProgressHook
import os

# Set page title
st.title("Speaker Diarization Web App")

# Hugging Face Token (Replace with your actual token)
HUGGINGFACE_ACCESS_TOKEN = "HUGGINGFACE_ACCESS_TOKEN_GOES_HERE"

# Set Hugging Face access token
os.environ["HF_HOME"] = HUGGINGFACE_ACCESS_TOKEN

# Load PyAnnote pipeline
@st.cache_resource
def load_pipeline():
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1")
    pipeline.to(torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    return pipeline

pipeline = load_pipeline()

# File uploader
uploaded_file = st.file_uploader("Upload an audio file", type=["wav", "mp3", "ogg"])

if uploaded_file is not None:
    # Save uploaded file temporarily
    file_path = os.path.join("temp_audio.wav")
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    st.audio(file_path, format='audio/wav')
    st.write("Processing audio file for speaker diarization...")
    
    # Run diarization
    with ProgressHook() as hook:
        diarization = pipeline(file_path, hook=hook)
    
    # Display results
    st.write("### Speaker Segments")
    diarization_result = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        diarization_result.append(f"start={turn.start:.1f}s stop={turn.end:.1f}s speaker_{speaker}")
    
    st.text("\n".join(diarization_result))

    # Clean up
    os.remove(file_path)
