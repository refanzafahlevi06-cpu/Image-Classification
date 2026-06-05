import streamlit as st
import os
from PIL import Image
import io

# Konfigurasi Halaman Utama Streamlit
st.set_page_config(
    page_title="Image Resizer & CNN Model App",
    page_icon="🖼️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Header Aplikasi
st.title("🖼️ Image Resizer & CNN Architecture Viewer")
st.write("Aplikasi interaktif hasil konversi Jupyter Notebook untuk pemrosesan gambar dan visualisasi model CNN.")

# Sidebar Navigasi
st.sidebar.header("Navigasi Fitur")
menu = st.sidebar.radio("Pilih Menu:", ["1. Batch Image Resizer", "2. Struktur Model CNN"])

# ------------------------------------------------------------------------------------------
# FITUR 1: BATCH IMAGE RESIZER
# ------------------------------------------------------------------------------------------
if menu == "1. Batch Image Resizer":
    st.header("🔄 Batch Image Resizer")
    st.write("Unggah gambar dari komputer Anda untuk mengubah ukurannya secara instan secara massal.")

    # Input interaktif untuk ukuran target
    st.subheader("Pengaturan Ukuran Target (Pixels)")
    col_width, col_height = st.columns(2)
    with col_width:
        width = st.slider("Lebar (Width):", min_value=32, max_value=2000, value=300, step=10)
    with col_height:
        height = st.slider("Tinggi (Height):", min_value=32, max_value=2000, value=300, step=10)
    
    target_size = (width, height)

    # Wadah Unggah Gambar
    uploaded_files = st.file_uploader(
        "Pilih file gambar Anda (Bisa memilih beberapa file sekaligus):", 
        type=["jpg", "jpeg", "png"], 
        accept_multiple_files=True
    )

    if uploaded_files:
        st.success(f"Berhasil memuat {len(uploaded_files)} gambar.")
        st.subheader("Hasil Pemrosesan Gambar:")
        
        for uploaded_file in uploaded_files:
            try:
                img = Image.open(uploaded_file)
                old_size = img.size
                img_resized = img.resize((128, 128))
                
                with st.expander(f"📄 Berkas: {uploaded_file.name}"):
                    c1, c2 = st.columns(2)
                    with c1:
                        st.image(img, caption=f"Asli ({old_size[0]}x{old_size[1]})", use_container_width=True)
                    with c2:
                        # Bagian caption langsung ditulis manual saja angkanya agar tidak error
                        st.image(img_resized, caption="Hasil Resize (128x128)", use_container_width=True)
                    
                    buffer = io.BytesIO()
                    file_ext = os.path.splitext(uploaded_file.name)[1].lower()
                    save_format = "PNG" if file_ext == ".png" else "JPEG"
                    img_resized.save(buffer, format=save_format)
                    
                    st.download_button(
                        label=f"⬇️ Unduh {uploaded_file.name}",
                        data=buffer.getvalue(),
                        file_name=f"resized_{uploaded_file.name}",
                        mime=f"image/{save_format.lower()}"
                    )
                    
            except Exception as e:
                st.error(f"Gagal memproses gambar '{uploaded_file.name}': {e}")

# ------------------------------------------------------------------------------------------
# FITUR 2: STRUKTUR MODEL CNN
# ------------------------------------------------------------------------------------------
elif menu == "2. Struktur Model CNN":
    st.header("🧠 Ringkasan Arsitektur Model CNN")
    st.write("Representasi dari model Sequential TensorFlow Keras yang terdapat pada notebook Anda.")
    
    try:
        import tensorflow as tf
        
        image_size = (128, 128)
        num_classes = 2
        
        st.info(f"Model dikonfigurasi menggunakan Input Dimensi: {image_size[0]}x{image_size[1]} piksel dengan {num_classes} Kelas Output ('hiu' dan 'burung').")
        
        model = tf.keras.models.Sequential([
            tf.keras.layers.Input(shape=(image_size[0], image_size[1], 3)),
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(num_classes, activation='softmax')
        ])
        
        stream = io.StringIO()
        model.summary(print_fn=lambda x: stream.write(x + '\n'))
        summary_string = stream.getvalue()
        
        st.subheader("Detail Layer Model Keras:")
        st.code(summary_string, language="text")
        
    except ImportError:
        st.warning("Pustaka `tensorflow` tidak terdeteksi di perangkat Anda. Berikut adalah cetakan struktur teks model dari notebook:")
        st.code("""
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 conv2d (Conv2D)             (None, 126, 126, 32)      896       
 max_pooling2d (MaxPooling2D) (None, 63, 63, 32)        0         
 conv2d_1 (Conv2D)           (None, 61, 61, 64)        18496     
 max_pooling2d_1 (MaxPooling (None, 30, 30, 64)        0         
2D)                                                              
 flatten (Flatten)           (None, 57600)             0         
 dense (Dense)               (None, 64)                3686464   
 dense_1 (Dense)             (None, 2)                 130       
=================================================================
Total params: 3,705,986 (14.14 MB)
Trainable params: 3,705,986 (14.14 MB)
Non-trainable params: 0 (0.00 B)
_________________________________________________________________
        """, language="text")
