col1, col2 = st.columns([1, 2])  # Kiri untuk kontrol, kanan untuk input dan hasil

with col1:
    st.image("logo-saraswanti.png", width=200)
    st.markdown("### Pilih metode input gambar:")
    option = st.radio("", ["Upload Gambar", "Gunakan Kamera"])

with col2:
    image = None

    if option == "Upload Gambar":
        uploaded_file = st.file_uploader("📂 Drag & drop gambar buah sawit di sini", type=["jpg", "jpeg", "png"])
        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, caption="Gambar yang diunggah", use_container_width=True)

    elif option == "Gunakan Kamera":
        st.markdown("### Gunakan Kamera Belakang (Environment)")

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
            st.session_state["camera_image"] = base64_img
            try:
                header, encoded = base64_img.split(",", 1)
                decoded = base64.b64decode(encoded)
                image = Image.open(BytesIO(decoded))
                st.image(image, caption="📷 Gambar dari Kamera", use_container_width=True)
            except Exception as e:
                st.error(f"Gagal memproses gambar dari kamera: {e}")

    # Proses Deteksi jika gambar tersedia
    if image:
        with st.spinner("🔍 Memproses gambar..."):
            model = load_model()
            results = predict_image(model, image)
            img_with_boxes, class_counts = draw_results(image, results)

            st.image(img_with_boxes, caption="📊 Hasil Deteksi", use_container_width=True)

            st.subheader("Jumlah Objek Terdeteksi:")
            for name, count in class_counts.items():
                st.write(f"- **{name}**: {count}")

            # Tombol download
            st.markdown(get_image_download_link(img_with_boxes), unsafe_allow_html=True)
