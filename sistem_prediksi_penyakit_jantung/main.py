import streamlit as st 
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import requests
from io import BytesIO

# ======================
# SETTING HALAMAN
# ======================
st.set_page_config(
    page_title="Prediksi Penyakit Jantung",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ======================
# LOAD ASSETS
# ======================
def load_assets():
    url = "https://raw.githubusercontent.com/Darmanto212/kelompok7/main/sistem_prediksi_penyakit_jantung/heart_image.jpg"
    try:
        response = requests.get(url)
        response.raise_for_status()  # Memastikan URL dapat diakses
        img = Image.open(BytesIO(response.content))  # Membuka gambar dari data yang diunduh
        return img
    except requests.exceptions.RequestException as e:
        st.error(f"Error mengunduh gambar: {e}")
        return None
    except Exception as e:
        st.error(f"Terjadi kesalahan saat memuat gambar: {e}")
        return None


# ======================
# LOAD MODEL & SCALER
# ======================
@st.cache_resource
def load_models():
    # URL model yang diunduh dari GitHub
    model_urls = {
        "Random Forest": "https://raw.githubusercontent.com/Darmanto212/kelompok7/main/sistem_prediksi_penyakit_jantung/random_forest_model.pkl",
        "SVM": "https://raw.githubusercontent.com/Darmanto212/kelompok7/main/sistem_prediksi_penyakit_jantung/svm_model.pkl",
        "Logistic Regression": "https://raw.githubusercontent.com/Darmanto212/kelompok7/main/sistem_prediksi_penyakit_jantung/logistic_regression_model.pkl"
    }

    models = {}
    for name, url in model_urls.items():
        try:
            response = requests.get(url)
            response.raise_for_status()
            models[name] = pickle.load(BytesIO(response.content))
        except requests.exceptions.RequestException as e:
            st.error(f"Error mengunduh model {name}: {e}")
        except pickle.UnpicklingError as e:
            st.error(f"Error saat memuat model {name}: {e}")
        except Exception as e:
            st.error(f"Terjadi kesalahan lainnya saat memuat model {name}: {e}")

    # URL untuk scaler
    scaler_url = "https://raw.githubusercontent.com/Darmanto212/kelompok7/main/sistem_prediksi_penyakit_jantung/scaler.pkl"
    scaler = None
    try:
        response = requests.get(scaler_url)
        response.raise_for_status()
        scaler = pickle.load(BytesIO(response.content))
    except requests.exceptions.RequestException as e:
        st.error(f"Error mengunduh scaler: {e}")
    except pickle.UnpicklingError as e:
        st.error(f"Error saat memuat scaler: {e}")
    except Exception as e:
        st.error(f"Terjadi kesalahan lainnya saat memuat scaler: {e}")

    return models, scaler


# ======================
# TAMPILAN UTAMA
# ======================
def main():
    # Memuat gambar
    heart_image = load_assets()
    if heart_image:
        st.sidebar.image(heart_image, width=250)
    else:
        st.sidebar.write("Gambar tidak dapat dimuat.")
    
    st.sidebar.title("Navigasi")
    menu = st.sidebar.radio("Pilih Menu:", ["Prediksi", "Tentang Aplikasi"])

    if menu == "Prediksi":
        show_prediction_page()
    else:
        show_about_page()

# ======================
# HALAMAN PREDIKSI
# ======================
def show_prediction_page():
    st.title("🫀 Prediksi Risiko Penyakit Jantung")
    st.write(
        """
    Aplikasi ini memprediksi risiko penyakit jantung menggunakan 3 algoritma Machine Learning.
    Silakan isi data pasien di bawah ini:
    """
    )

    with st.expander("📋 Form Input Data Pasien", expanded=True):
        col1, col2 = st.columns(2)

        with col1:
            age = st.slider("Usia (age)", 20, 100, 55)
            sex = st.radio("Jenis Kelamin (sex)", ["0. Perempuan", "1. Laki-laki"])
            cp = st.selectbox(
                "Jenis Nyeri Dada (cp)",
                [
                    "0. Typical angina",
                    "1. Atypical angina",
                    "2. Non-anginal pain",
                    "3. Asymptomatic",
                ],
            )
            trestbps = st.slider("Tekanan Darah mmHg (trestbps)", 90, 200, 120)
            chol = st.slider("Kolesterol mg/dl (chol)", 100, 600, 250)
            fbs = st.radio("Gula Darah Puasa > 120 mg/dl (fbs)", ["0. Tidak", "1. Ya"])

        with col2:
            restecg = st.selectbox(
                "Hasil Elektrokardiografi (restecg)",
                [
                    "0. Normal",
                    "1. Kelainan Gelombang ST-T",
                    "2. Hipertrofi Ventrikel Kiri",
                ],
            )
            thalach = st.slider("Denyut Jantung Maksimal (thalach)", 70, 220, 150)
            exang = st.radio(
                "Nyeri Dada Dipicu Olahraga (exang)", ["0. Tidak", "1. Ya"]
            )
            oldpeak = st.slider("Depresi ST (oldpeak)", 0.0, 6.2, 1.0)
            slope = st.selectbox(
                "Slope Segmen ST (slope)", ["0. Upsloping", "1. Flat", "2. Downsloping"]
            )
            ca = st.selectbox(
                "Jumlah Pembuluh Darah Utama yang Terlihat (ca)",
                options=[0, 1, 2, 3],
                index=0,
                help="Nilai antara 0-3 dari hasil fluoroskopi",
            )
            thal = st.selectbox(
                "Hasil Thallium Scan (thal)",
                ["1. Normal", "2. Cacat Tetap", "3. Cacat Reversibel"],
            )
    
    # Konversi input ke numerik
    sex = 1 if sex == "1. Laki-laki" else 0
    cp_mapping = {
        "0. Typical angina": 0,
        "1. Atypical angina": 1,
        "2. Non-anginal pain": 2,
        "3. Asymptomatic": 3,
    }
    cp = cp_mapping[cp]
    fbs = 1 if fbs == "1. Ya" else 0
    restecg_mapping = {
        "0. Normal": 0,
        "1. Kelainan Gelombang ST-T": 1,
        "2. Hipertrofi Ventrikel Kiri": 2,
    }
    restecg = restecg_mapping[restecg]
    exang = 1 if exang == "1. Ya" else 0
    slope_mapping = {"0. Upsloping": 0, "1. Flat": 1, "2. Downsloping": 2}
    slope = slope_mapping[slope]
    thal_mapping = {"1. Normal": 1, "2. Cacat Tetap": 2, "3. Cacat Reversibel": 3}
    thal = thal_mapping[thal]

    # Tombol prediksi
    if st.button(" Prediksi Sekarang", use_container_width=True):
        with st.spinner("Menganalisis data..."):
            # Format input data
            input_data = np.array(
                [
                    [
                        age,
                        sex,
                        cp,
                        trestbps,
                        chol,
                        fbs,
                        restecg,
                        thalach,
                        exang,
                        oldpeak,
                        slope,
                        ca,
                        thal,
                    ]
                ]
            )

            # Cek apakah scaler ada dan bisa dipakai
            models, scaler = load_models()
            if scaler is None:
                st.error("Scaler tidak dapat dimuat. Coba lagi nanti.")
                return

            # Scaling
            try:
                input_scaled = scaler.transform(input_data)
            except Exception as e:
                st.error(f"Terjadi kesalahan saat scaling data: {e}")
                return

            # Prediksi
            results = []
            for name, model in models.items():
                try:
                    proba = model.predict_proba(input_scaled)[0][1]
                    prediction = (
                        "Positif Penyakit Jantung"
                        if proba >= 0.5
                        else "Negatif Penyakit Jantung"
                    )
                    results.append(
                        {
                            "Model": name,
                            "Hasil Prediksi": prediction,
                            "Tingkat Risiko": f"{proba*100:.1f}%",
                            "Keterangan": (
                                "Risiko Tinggi" if proba >= 0.5 else "Risiko Rendah"
                            ),
                        }
                    )
                except Exception as e:
                    st.error(f"Terjadi kesalahan saat melakukan prediksi dengan model {name}: {e}")
                    return

            # Simpan hasil prediksi ke session state
            st.session_state.results = results

            # Tampilkan hasil
            st.success("Analisis Selesai!")

            col1, col2 = st.columns(2)

            with col1:
                st.subheader("📊 Hasil Prediksi")
                results_df = pd.DataFrame(results)
                st.dataframe(
                    results_df.style.applymap(
                        lambda x: (
                            "background-color: #820901"
                            if "Positif" in str(x)
                            else "background-color: #008000"
                        ),
                        subset=["Hasil Prediksi"],
                    ),
                    hide_index=True,
                )

            with col2:
                st.subheader("📈 Visualisasi Risiko")
                fig, ax = plt.subplots()
                colors = [
                    "#ff6b6b" if float(x["Tingkat Risiko"][:-1]) >= 50 else "#51cf66"
                    for x in results
                ]
                ax.bar(
                    results_df["Model"],
                    results_df["Tingkat Risiko"].str.replace("%", "").astype(float),
                    color=colors,
                )
                ax.axhline(y=50, color="red", linestyle="--")
                ax.set_ylabel("Persentase Risiko (%)")
                ax.set_ylim(0, 100)
                plt.xticks(rotation=45)
                st.pyplot(fig)

            # Rekomendasi
            st.subheader("💡 Rekomendasi")
            if any(float(x["Tingkat Risiko"][:-1]) >= 50 for x in results):
                st.warning(
                    """
                **Pasien memiliki risiko penyakit jantung. Disarankan untuk:**
                - Konsultasi segera dengan dokter spesialis jantung
                - Melakukan pemeriksaan EKG dan tes treadmill
                - Memperbaiki pola makan dan gaya hidup
                """
                )
            else:
                st.info(
                    """
                **Pasien memiliki risiko rendah. Tetap jaga kesehatan dengan:**
                - Rutin berolahraga
                - Diet seimbang
                - Cek kesehatan berkala
                """
                )


# ======================
# HALAMAN TENTANG
# ======================
def show_about_page():
    st.title("📚 Tentang Aplikasi Ini")

    st.markdown(
        """
    ### ❤️ Aplikasi Prediksi Penyakit Jantung
    Aplikasi ini menggunakan 3 model Machine Learning untuk memprediksi risiko penyakit jantung 
    berdasarkan data klinis pasien.
    """
    )

    with st.expander("🔍 Detail Model", expanded=False):
        st.markdown(
            """
        **Model yang digunakan:**
        1. Random Forest Classifier
        2. Support Vector Machine (SVM)
        3. Logistic Regression
        
        **Akurasi Model:**
        - Random Forest: 84%
        - SVM: 84% 
        - Logistic Regression: 86%
        """
        )

    with st.expander("📝 Petunjuk Penggunaan", expanded=False):
        st.markdown(
            """
        1. Isi semua data pasien pada form input
        2. Klik tombol "Prediksi Sekarang"
        3. Lihat hasil prediksi dan rekomendasi
        4. Untuk prediksi baru, refresh halaman
        """
        )

    st.markdown(
        """
    ### ⚠️ Disclaimer
    Hasil prediksi ini bukan diagnosis medis. Konsultasikan dengan dokter untuk pemeriksaan lebih lanjut.
    """
    )


# ======================
# RUN APLIKASI
# ======================
if __name__ == "__main__":
    main()
