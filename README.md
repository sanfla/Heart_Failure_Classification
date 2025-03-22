# Prediksi Gagal Jantung

## Deskripsi
Proyek ini bertujuan untuk memprediksi risiko gagal jantung berdasarkan data medis pasien menggunakan model pembelajaran mesin. Aplikasi ini telah **dideploy dengan Streamlit** agar pengguna dapat dengan mudah melakukan prediksi melalui antarmuka web.

## Dataset
Dataset yang digunakan mencakup berbagai fitur medis seperti:
- Usia pasien
- Tekanan darah
- Kadar kolesterol
- Detak jantung
- Faktor risiko lainnya

## Instalasi
```bash
pip install -r requirements.txt
```

## Penggunaan
```bash
git clone https://github.com/sanfla/Heart_Failure_Classification.git
cd Heart_Failure_Classification
python scripts/train.py  # Melatih model
python scripts/predict.py --input data_sample.csv  # Prediksi data baru
streamlit run app.py  # Menjalankan aplikasi
```

## Struktur Direktori
```
Heart_Failure_Classification/
│── data/
│── models/
│── notebooks/
│── scripts/
│── app.py
│── requirements.txt
│── README.md
```
