import streamlit as st
from PIL import Image
import cv2
import numpy as np
from collections import Counter
from ultralytics import YOLO
from supervision import BoxAnnotator, LabelAnnotator, Color, Detections
from io import BytesIO

# Konfigurasi halaman
st.set_page_config(page_title="Deteksi Buah Sawit", layout="wide")

# Load model YOLO
@st.cache_resource
def load_model():
    return YOLO("best.pt")  # Ganti dengan path model kamu

model = load_model()

# Warna label
label_to_color = {
    "Masak": Color.RED,
    "Mengkal": Color.YELLOW,
    "Mentah": Color.BLACK
}
label_annotator = LabelAnnotator()

# Fungsi anotasi
def draw_results(image, results):
    img = np.array(image.convert("RGB"))
    class_counts = Counter()

    for result in results:
        boxes = result.boxes
        names = result.names

        xyxy = boxes.xyxy.cpu().numpy()
        class_ids = boxes.cls.cpu().numpy().astype(int)
        confidences = boxes.conf.cpu().numpy()

        for box, class_id, conf in zip(xyxy, class_ids, confidences):
            class_name = names[class_id]
            label = f"{class_name}: {conf:.2f}"
            color = label_to_color.get(class_name, Color.WHITE)

            class_counts[class_name] += 1

            box_annotator = BoxAnnotator(color=color)
            detection = Detections(
                xyxy=np.array([box]),
                confidence=np.array([conf]),
                class_id=np.array([class_id])
            )

            img = box_annotator.annotate(scene=img, detections=detection)
            img = label_annotator.annotate(scene=img, detections=detection, labels=[label])

    return Image.fromarray(img), class_counts

# Sidebar
# Sidebar
with st.sidebar:
    st.markdown("<div style='text-align:center;'>", unsafe_allow_html=True)
    st.image("logo-saraswanti.png", width=150)
    st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown(
        """
        <div style="margin-top:20px;">
            <h4>Pilih metode input gambar:</h4>
        </div>
        """, 
        unsafe_allow_html=True
    )

    image = None
    option = st.radio("", ["Upload Gambar", "Gunakan Kamera"])

    if option == "Upload Gambar":
        st.markdown("<p style='font-size:16px; font-weight:bold;'>Unggah gambar</p>", unsafe_allow_html=True)
        uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"])
        if uploaded_file:
            image = Image.open(uploaded_file)

    elif option == "Gunakan Kamera":
        st.markdown("<p style='font-size:16px; font-weight:bold;'>Ambil gambar dengan kamera</p>", unsafe_allow_html=True)
        camera_photo = st.camera_input("")
        if camera_photo is not None:
            image = Image.open(camera_photo)

# Judul dan deskripsi
st.markdown("<h1 style='text-align:center;'>🌴 Deteksi dan Klasifikasi Kematangan Buah Sawit</h1>", unsafe_allow_html=True)

st.markdown("""
<div style="text-align:center; font-size:16px; max-width:800px; margin:auto;">
    Sistem ini menggunakan teknologi YOLO untuk mendeteksi dan mengklasifikasikan kematangan buah kelapa sawit 
    secara otomatis berdasarkan gambar input. Dengan deteksi yang akurat, diharapkan dapat membantu dalam 
    pengelolaan perkebunan kelapa sawit yang lebih efisien dan hasil panen yang optimal.
</div>
""", unsafe_allow_html=True)

# Jika ada gambar input
if image:
    with st.spinner("🔍 Memproses gambar..."):
        results = model(image)
        result_img, class_counts = draw_results(image, results)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### 🖼️ Gambar Input")
            st.image(image, use_container_width=True)

        with col2:
            st.markdown("### 📊 Hasil Deteksi")
            st.image(result_img, use_container_width=True)

        st.subheader("Jumlah Objek Terdeteksi:")
        for name, count in class_counts.items():
            st.write(f"- **{name}**: {count}")

        # Tombol download hasil
        buffered = BytesIO()
        result_img.save(buffered, format="PNG")
        img_bytes = buffered.getvalue()

        st.download_button(
            label="⬇️ Download Gambar Berlabel",
            data=img_bytes,
            file_name="hasil_deteksi.png",
            mime="image/png"
        )
else:
    st.info("Silakan unggah gambar atau ambil foto dengan kamera untuk memulai deteksi.")
