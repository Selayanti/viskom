import streamlit as st
import cv2
import numpy as np
from PIL import Image
from collections import Counter
import base64
from io import BytesIO
from ultralytics import YOLO
from supervision import BoxAnnotator, LabelAnnotator, Color, Detections

# === Konfigurasi halaman ===
st.set_page_config(page_title="Deteksi Buah Sawit", layout="wide")

# === Load model YOLO ===
@st.cache_resource
def load_model():
    return YOLO("best.pt")  # ganti path model Anda

model = load_model()

# === Warna untuk label ===
label_to_color = {
    "Masak": Color.RED,
    "Mengkal": Color.YELLOW,
    "Mentah": Color.BLACK
}
label_annotator = LabelAnnotator()

# === Fungsi anotasi gambar ===
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

# === Sidebar (panel kiri/slide) ===
with st.sidebar:
    st.image("logo-saraswanti.png", width=150)
    st.markdown("### Pilih metode input gambar:")
    option = st.radio("", ["Upload Gambar", "Gunakan Kamera"])

    image = None

    if option == "Upload Gambar":
        uploaded_file = st.file_uploader("Unggah gambar", type=["jpg", "jpeg", "png"])
        if uploaded_file:
            image = Image.open(uploaded_file)

    elif option == "Gunakan Kamera":
        st.markdown("### Kamera Belakang (Environment)")

        camera_html = """
        <div style="text-align:center;">
            <video id="video" autoplay playsinline style="width:100%; border:1px solid gray;"></video>
            <br/>
            <button onclick="takePhoto()" style="margin-top:10px; padding:10px 20px;">📸 Ambil Gambar</button>
            <canvas id="canvas" style="display:none;"></canvas>
        </div>

        <script>
            async function startCamera() {
                try {
                    const stream = await navigator.mediaDevices.getUserMedia({
                        video: { facingMode: { ideal: "environment" } },
                        audio: false
                    });
                    const video = document.getElementById('video');
                    video.srcObject = stream;
                } catch (err) {
                    alert("Gagal mengakses kamera: " + err.message);
                }
            }

            function takePhoto() {
                const video = document.getElementById('video');
                const canvas = document.getElementById('canvas');
                const context = canvas.getContext('2d');
                canvas.width = video.videoWidth;
                canvas.height = video.videoHeight;
                context.drawImage(video, 0, 0, canvas.width, canvas.height);
                const dataURL = canvas.toDataURL('image/png');

                const input = window.parent.document.querySelector('input[data-testid="stTextInput"]');
                if (input) {
                    input.value = dataURL;
                    input.dispatchEvent(new Event("input", { bubbles: true }));
                }
            }

            document.addEventListener("DOMContentLoaded", startCamera);
        </script>
        """

        st.components.v1.html(camera_html, height=600)

        base64_img = st.text_input("Gambar dari Kamera (tersembunyi)", type="default", label_visibility="collapsed")
        if base64_img.startswith("data:image"):
            try:
                header, encoded = base64_img.split(",", 1)
                decoded = base64.b64decode(encoded)
                image = Image.open(BytesIO(decoded))
            except Exception as e:
                st.error(f"Gagal memproses gambar dari kamera: {e}")

# === Area utama (konten utama di kanan) ===
st.title("📷 Deteksi dan Klasifikasi Kematangan Buah Sawit")

if image:
    st.image(image, caption="🖼️ Gambar Input", use_container_width=True)

    with st.spinner("🔍 Memproses gambar..."):
        results = model(image)
        result_img, class_counts = draw_results(image, results)

        st.image(result_img, caption="📊 Hasil Deteksi", use_container_width=True)
        st.subheader("Jumlah Objek Terdeteksi:")
        for name, count in class_counts.items():
            st.write(f"- **{name}**: {count}")

        # Tombol download
        buffered = BytesIO()
        result_img.save(buffered, format="PNG")
        img_bytes = buffered.getvalue()

        st.download_button(
            label="⬇️ Download Gambar Berlabel",
            data=img_bytes,
            file_name="hasil_deteksi.png",
            mime="image/png"
        )
