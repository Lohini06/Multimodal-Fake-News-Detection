import streamlit as st
import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PIL import Image
from src.utils.config import load_config
from src.models.clip_classifier import MultimodalClassifier

st.set_page_config(
    page_title="Fake News Detector",
    page_icon="🔍",
    layout="centered"
)

@st.cache_resource
def load_model():
    config = load_config("config.yaml")
    model = MultimodalClassifier(config)
    checkpoint_path = os.path.join(
        config["outputs"]["checkpoint_dir"],
        config["outputs"]["best_model_name"]
    )
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, config

st.title("Multimodal Fake News Detector")
st.markdown("Detects fake news by analyzing both **headline text** and **image** together using CLIP.")
st.divider()

headline = st.text_area("News Headline", placeholder="Enter the news headline here...", height=100)
uploaded_image = st.file_uploader("Upload News Image (optional)", type=["jpg", "jpeg", "png"])

col1, col2 = st.columns(2)
with col1:
    if uploaded_image:
        image = Image.open(uploaded_image).convert("RGB")
        st.image(image, caption="Uploaded image", use_column_width=True)
    else:
        st.info("No image uploaded — using blank placeholder.")

with col2:
    if st.button("Analyze", use_container_width=True, type="primary"):
        if not headline.strip():
            st.error("Please enter a headline!")
        else:
            with st.spinner("Analyzing..."):
                model, config = load_model()

                if uploaded_image:
                    image = Image.open(uploaded_image).convert("RGB")
                else:
                    image = Image.new("RGB", (224, 224), color=(255, 255, 255))

                image_tensor = model.preprocess(image).unsqueeze(0)

                with torch.no_grad():
                    logits = model(image_tensor, [headline])
                    probs = torch.softmax(logits, dim=1)[0]
                    pred = logits.argmax(dim=1).item()

                real_prob = round(probs[0].item() * 100, 1)
                fake_prob = round(probs[1].item() * 100, 1)

                st.divider()
                if pred == 1:
                    st.error("FAKE NEWS DETECTED")
                else:
                    st.success("LIKELY REAL NEWS")

                st.metric("Real probability", f"{real_prob}%")
                st.metric("Fake probability", f"{fake_prob}%")

st.divider()
st.markdown("Built with CLIP + PyTorch | Multimodal Fake News Detection")