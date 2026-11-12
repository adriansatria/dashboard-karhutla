# Dashboard Risiko Karhutla

Dashboard Streamlit yang menampilkan hasil notebook `karhutla_eda_pemodelan.ipynb`:
peta risiko per grid 0,5°, skenario fase El Niño, perbandingan prediksi dengan kejadian aktual,
dan evaluasi model. Aplikasi hanya membaca berkas hasil; tidak mengunduh data dan tidak melatih model,
sehingga tidak membutuhkan backend terpisah.

## Struktur

```
dashboard_karhutla/
├── app.py               aplikasi
├── requirements.txt     dependensi (tanpa scikit-learn dan xgboost)
├── data/                taruh hasil notebook di sini
└── data_contoh/         data TIRUAN untuk mencoba tampilan
```

## Langkah 1: jalankan notebook

Jalankan `karhutla_eda_pemodelan.ipynb` (Google Colab atau lokal) sampai sel terakhir.
Notebook membuat folder `output_dashboard/`. Unduh folder itu dari Colab (klik kanan, Download, atau zip dulu).

## Langkah 2: salin hasil ke folder `data/`

Salin berkas berikut dari `output_dashboard/` ke `data/`:

```
grid.geojson               info_sel.csv               metrik.json
prediksi_riwayat.csv       prediksi_bulan_depan.csv   skenario_enso.csv
perbandingan_model.csv     feature_importance.csv     ablasi_fitur.csv
validasi_el_nino.csv
```

Dua berkas tidak diperlukan dan sebaiknya jangan disalin: `dataset_grid_bulan.csv.gz` (besar) dan `model_terbaik.joblib`.

## Langkah 3: jalankan lokal

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Buka alamat yang tampil di terminal (biasanya http://localhost:8501).
Jika folder `data/` masih kosong, aplikasi otomatis memakai `data_contoh/` dengan peringatan data tiruan.

## Langkah 4: pasang online (opsional)

1. Unggah folder ini ke repositori GitHub, termasuk isi `data/`.
2. Buka https://share.streamlit.io, login dengan GitHub, pilih **Create app**.
3. Isi repositori, branch, dan `app.py` sebagai berkas utama, lalu **Deploy**.

Hasilnya berupa tautan yang bisa dibagikan ke dosen.

## Memperbarui hasil

Jalankan ulang notebook, ganti isi `data/`, lalu commit. Aplikasi di Streamlit Cloud ikut diperbarui otomatis.

## Pengaturan lain

- `KARHUTLA_DATA`: lokasi folder data lain.
- `KARHUTLA_MAP_STYLE`: gaya peta dasar. Bawaan `carto-positron` (membutuhkan internet). Gunakan `white-bg` jika tanpa internet.

## Catatan

- Skenario El Niño memakai frekuensi historis per fase, bukan mengubah ONI pada model.
- Titik panas adalah indikator, bukan kebakaran yang terkonfirmasi.
