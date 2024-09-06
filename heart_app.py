import streamlit as st
import pickle
import pandas as pd
from PIL import Image
import requests
from io import BytesIO

page = st.sidebar.selectbox("Pilih Halaman", ["Pendahuluan", "Perhitungan Prediksi"])

model_url = "https://github.com/sanfla/Heart_Failure_Classification/raw/main/model.pkl"
response = requests.get(model_url)
model = pickle.load(BytesIO(response.content))

if page == "Pendahuluan":

    st.image('https://github.com/sanfla/Heart_Failure_Classification/blob/main/HF_image.png?raw=true', use_column_width=True)

    st.title("Pendahuluan")
    data = pd.read_csv("https://raw.githubusercontent.com/sanfla/Heart_Failure_Classification/main/heart_failure.csv")
    st.dataframe(data.head(5))

    st.write("""
    Penyakit kardiovaskular (CVD) adalah penyebab utama kematian di seluruh dunia, 
    dengan perkiraan 17,9 juta nyawa hilang setiap tahun, yang menyumbang 31% dari 
    semua kematian global. Empat dari lima kematian akibat CVD disebabkan oleh serangan 
    jantung dan stroke, dan sepertiga dari kematian ini terjadi secara prematur pada orang 
    di bawah usia 70 tahun. Kegagalan jantung adalah kejadian umum yang disebabkan oleh CVD, 
    dan dataset ini berisi 11 fitur yang dapat digunakan untuk memprediksi kemungkinan penyakit 
    jantung.

    Orang dengan penyakit kardiovaskular atau yang berisiko tinggi kardiovaskular (karena 
    adanya satu atau lebih faktor risiko seperti hipertensi, diabetes, hiperlipidemia, atau 
    penyakit yang sudah ada) memerlukan deteksi dan manajemen dini di mana model pembelajaran 
    mesin dapat sangat membantu.

    **Fitur**

    - **Age**: usia pasien [tahun]
    - **Sex**: jenis kelamin pasien [M: Male, F: Female]
    - **ChestPainType**: jenis nyeri dada [TA: Typical Angina, ATA: Atypical Angina, NAP: Non-Anginal Pain, ASY: Asymptomatic]
    - **RestingBP**: tekanan darah istirahat [mm Hg]
    - **Cholesterol**: kolesterol serum [mm/dl]
    - **FastingBS**: gula darah puasa [1: jika FastingBS > 120 mg/dl, 0: sebaliknya]
    - **RestingECG**: hasil elektrokardiogram istirahat [Normal: Normal, ST: memiliki kelainan gelombang ST-T (terbaliknya gelombang 
    T dan/atau elevasi atau depresi ST > 0,05 mV), LVH: menunjukkan kemungkinan atau defenitif hipertrofi ventrikel kiri menurut kriteria Estes]
    - **MaxHR**: detak jantung maksimum yang dicapai [Nilai numerik antara 60 dan 202]
    - **ExerciseAngina**: angina yang diinduksi olahraga [Y: Ya, N: Tidak]
    - **Oldpeak**: oldpeak = ST [Nilai numerik diukur dalam depresi]
    - **ST_Slope**: kemiringan segmen ST puncak olahraga [Up: menaik, Flat: datar, Down: menurun]
    - **HeartDisease**: kelas output [1: penyakit jantung, 0: Normal]
    """)


elif page == "Perhitungan Prediksi":

    st.image('https://github.com/sanfla/Heart_Failure_Classification/blob/main/HF_image.png?raw=true', use_column_width=True)

    st.title("Perhitungan Prediksi")

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input('Umur', min_value=30, max_value=75)
        sex = st.selectbox('Jenis Kelamin', options=['Male', 'Female'])
        chest_pain = st.selectbox('Tipe Nyeri Dada', options=['Typical Angina', 'Atypical Angina', 'Non-Anginal Pain', 'Asymptomatic'])

    with col2:
        cholesterol = st.number_input('Kadar Kolesterol (mg/dL)', min_value=0, max_value=600, value=200)
        fasting_bs = st.selectbox('Kadar Gula Darah Puasa', options=['True', 'False'])
        max_hr = st.number_input('Detak Jantung Maksimum', min_value=60, max_value=200, value=140)

    with col3:
        exercise_angina = st.selectbox('Angina saat Olahraga', options=['Yes', 'No'])
        oldpeak = st.number_input('Oldpeak', min_value=-10.0, max_value=10.0, value=0.0, step=0.1, format="%.1f")
        st_slope = st.selectbox('Kemiringan Segmen ST', options=['Up', 'Flat', 'Down'])

    sex = 1 if sex == 'Male' else 0
    chest_pain_type = {'Asymptomatic': 0, 'Atypical Angina': 1, 'Non-Anginal Pain': 2, 'Typical Angina': 3}[chest_pain]
    fasting_bs = 1 if fasting_bs == 'True' else 0
    exercise_angina = 1 if exercise_angina == 'Yes' else 0
    st_slope = {'Down': 0, 'Flat': 1, 'Up': 2}[st_slope]

    input_data = pd.DataFrame({
        'Age': [age],
        'Cholesterol': [cholesterol],
        'MaxHR': [max_hr],
        'Oldpeak': [oldpeak],
        'Sex': [sex],
        'ChestPainType': [chest_pain_type],
        'FastingBS': [fasting_bs],
        'ExerciseAngina': [exercise_angina],
        'ST_Slope': [st_slope]
    })

    if st.button('Predict'):
        prediction = model.predict(input_data)
        if prediction[0] == 1:
            st.subheader('Model memprediksi bahwa pasien **memiliki** penyakit jantung.')
        else:
            st.subheader('Model memprediksi bahwa pasien **tidak memiliki** penyakit jantung.')

