import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import torchvision.models as models

# ----------------------------
# Load Model
# ----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.resnet18(weights=None)
model.fc = nn.Sequential(
    nn.Dropout(0.3),
    nn.Linear(model.fc.in_features, 2)
)

model.load_state_dict(torch.load("model/pneumonia_model.pth", map_location=device))
model = model.to(device)
model.eval()

# ----------------------------
# Image Transform
# ----------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

classes = ["Normal", "Pneumonia"]

# ----------------------------
# Streamlit App
# ----------------------------
st.set_page_config(page_title="PneumoScan", page_icon="🫁", layout="centered")

st.title("🫁 PneumoScan")
st.subheader("AI-Powered Chest X-Ray Analysis")
st.write("Upload a chest X-ray image to check for signs of pneumonia.")

uploaded_file = st.file_uploader("Choose an X-ray image", type=["jpg", "jpeg", "png", "bmp"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded X-Ray", use_container_width=True)

    with st.spinner("Analyzing..."):
        img = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(img)
            probs = torch.softmax(outputs, dim=1)
            confidence, pred = torch.max(probs, 1)

        result = classes[pred.item()]
        conf = confidence.item() * 100
        normal_prob = probs[0][0].item() * 100
        pneumonia_prob = probs[0][1].item() * 100

    # --- Results ---
    st.markdown("---")

    if result == "Normal":
        st.success(f"**Prediction: {result}** ({conf:.1f}% confidence)")
    else:
        st.error(f"**Prediction: {result}** ({conf:.1f}% confidence)")

    # Probability breakdown
    st.markdown("#### Probability Breakdown")
    col1, col2 = st.columns(2)
    col1.metric("Normal", f"{normal_prob:.1f}%")
    col2.metric("Pneumonia", f"{pneumonia_prob:.1f}%")

    st.progress(normal_prob / 100, text=f"Normal: {normal_prob:.1f}%")
    st.progress(pneumonia_prob / 100, text=f"Pneumonia: {pneumonia_prob:.1f}%")

    # Low confidence warning
    if conf < 70:
        st.warning("⚠️ Low confidence prediction. This may not be a valid chest X-ray, "
                   "or the image quality is poor. Please consult a medical professional.")

    st.markdown("---")
    st.caption("**Disclaimer:** This tool is for educational purposes only. "
               "It is not a substitute for professional medical diagnosis. "
               "Always consult a qualified healthcare provider.")