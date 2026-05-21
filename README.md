# Dashboard Data Mining Ojek Online

Aplikasi web berbasis Python dan Streamlit untuk menganalisis pola permintaan layanan ojek online berdasarkan waktu, lokasi, dan jenis layanan menggunakan metode Apriori atau Association Rule Mining.

## Fitur

- Upload dataset CSV/XLSX dari user.
- Preview dataset, jumlah baris, jumlah kolom, missing values, tipe data, dan daftar nama kolom.
- Mapping kolom dinamis melalui selectbox untuk waktu, tanggal, layanan, pembayaran, jarak, tarif, kecamatan asal, kota/kabupaten asal, kecamatan tujuan, kota/kabupaten tujuan, sub layanan, jumlah titik pengambilan, dan jumlah titik pengantaran.
- Preprocessing otomatis: trim whitespace, normalisasi label kategori, hapus duplikasi, dan handle missing values.
- Kategori waktu otomatis: Pagi, Siang, Sore, Malam, dan Dini Hari.
- Analisis deskriptif dengan visualisasi Plotly.
- Insight jam ramai, lokasi asal/tujuan ramai, rute tersibuk, dan heatmap jam x lokasi asal berbasis gabungan kecamatan dan kota/kabupaten.
- Analisis pola lanjutan: waktu-layanan, waktu-sub layanan, lokasi-layanan, pembayaran-layanan, grouped order dari sub layanan, jumlah titik pengambilan/pengantaran, Ramadan vs pasca Ramadan 2026, tunai/non-tunai, dan pola historis pribadi.
- Apriori menggunakan `mlxtend`: one hot encoding, frequent itemsets, dan association rules.
- Filter interaktif berdasarkan tanggal, layanan, kategori waktu, pembayaran, lokasi asal gabungan, dan lokasi tujuan gabungan.
- Visualisasi hasil Apriori: top rules, frequent itemset chart, network graph, dan confidence vs lift.
- Desain responsif yang mengikuti tema perangkat/browser, termasuk sidebar menu tanpa radio button.

## Struktur Folder

```text
.
+-- app.py
+-- data/
+-- pages/
|   +-- 01_Panduan_Aplikasi.py
+-- utils/
|   +-- apriori_utils.py
|   +-- advanced_analysis.py
|   +-- data_loader.py
|   +-- mapping.py
|   +-- preprocessing.py
|   +-- transactions.py
|   +-- ui.py
|   +-- visualizations.py
+-- requirements.txt
```

## Cara Menjalankan Aplikasi

1. Buat virtual environment jika diperlukan.

```bash
python -m venv .venv
```

2. Aktifkan virtual environment.

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

3. Install dependencies.

```bash
pip install -r requirements.txt
```

4. Jalankan Streamlit.

```bash
streamlit run app.py
```

5. Buka URL lokal yang ditampilkan Streamlit, biasanya `http://localhost:8501`.

## Catatan Dataset

Aplikasi ini tidak melakukan hardcode nama kolom dataset. Setelah upload, user memilih kolom yang sesuai melalui fitur mapping. Untuk lokasi, kecamatan dapat digabung dengan kota/kabupaten menjadi format seperti `Kebon Jeruk, Jakarta Barat`. Kolom yang tidak tersedia dapat dibiarkan sebagai `-- Tidak digunakan --`, tetapi hasil analisis akan lebih lengkap jika kolom-kolom utama dipetakan.
