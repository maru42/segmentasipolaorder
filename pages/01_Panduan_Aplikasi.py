import streamlit as st


st.set_page_config(page_title="Panduan Aplikasi", layout="wide")

st.title("Panduan Menjalankan Dashboard")

st.markdown(
    """
    Dashboard ini dibuat untuk menganalisis pola permintaan layanan ojek online
    menggunakan metode Apriori atau Association Rule Mining.

    Alur penggunaan:

    1. Upload dataset CSV/XLSX dari menu utama.
    2. Periksa daftar kolom, jumlah baris, missing values, dan tipe data.
    3. Buka section mapping kolom, lalu sesuaikan kolom waktu, tanggal, layanan,
       pembayaran, jarak, tarif, kecamatan asal, kota/kabupaten asal,
       kecamatan tujuan, kota/kabupaten tujuan, dan sub layanan.
    4. Jalankan preprocessing.
    5. Gunakan filter interaktif bila ingin menganalisis subset data.
    6. Atur minimum support, confidence, dan lift.
    7. Baca frequent itemset, association rules, dan visualisasi network graph.
    8. Gunakan Analisis Pola Lanjutan untuk melihat hubungan waktu-layanan,
       lokasi-layanan, pembayaran-layanan, grouped order, Ramadan 2026,
       tunai/non-tunai, dan pola historis pribadi.

    Aplikasi tidak mengharuskan nama kolom tertentu. Semua proses memakai mapping
    yang dipilih user setelah dataset diupload.
    """
)
