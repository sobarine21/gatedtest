import streamlit as st
import torch
from pyannote.audio import Pipeline
from pyannote.audio.pipelines.utils.hook import ProgressHook
import os

# Set page title
st.title("Speaker Diarization Web App")

# Hugging Face Token (Replace with your actual token)
HUGGINGFACE_ACCESS_TOKEN = "HUGGINGFACE_ACCESS_TOKEN_GOES_HERE"

# Set Hugging Face access token environment variable
os.environ["HF_HOME"] = HUGGINGFACE_ACCESS_TOKEN

# Load PyAnnote pipeline
@st.cache_resource
def load_pipeline():
    try:
        # Load the pipeline from Hugging Face
        pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1")
        
        # Check if pipeline loaded successfully
        if pipeline is None:
            st.error("Failed to load the speaker diarization model.")
            return None
        
        # Force CPU for debugging, change this to "cuda" if you want to use GPU
        device = torch.device("cpu")  # Force CPU usage for debugging
        pipeline.to(device)
        
        return pipeline
    except Exception as e:
        st.error(f"Error loading the pipeline: {e}")
        return None

# Load the pipeline
pipeline = load_pipeline()

# Check if the pipeline was loaded successfully
if pipeline is None:
    st.error("Pipeline loading failed. Please check the Hugging Face token or internet connection.")
else:
    # File uploader for audio files
    uploaded_file = st.file_uploader("Upload an audio file", type=["wav", "mp3", "ogg"])

    if uploaded_file is not None:
        # Save uploaded file temporarily
        file_path = os.path.join("temp_audio.wav")
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Display uploaded audio file
        st.audio(file_path, format='audio/wav')
        st.write("Processing audio file for speaker diarization...")

        # Run diarization with a progress hook
        with ProgressHook() as hook:
            diarization = pipeline(file_path, hook=hook)

        # Display diarization results
        st.write("### Speaker Segments")
        diarization_result = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            diarization_result.append(f"start={turn.start:.1f}s stop={turn.end:.1f}s speaker_{speaker}")

        st.text("\n".join(diarization_result))

        # Clean up the temporary file
        os.remove(file_path)

