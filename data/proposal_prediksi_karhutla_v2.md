# PROPOSAL TUGAS BESAR PROYEK SAINS DATA

## Prediksi Wilayah Rawan Kebakaran Hutan dan Lahan Satu Bulan ke Depan Menggunakan Machine Learning Berbasis Titik Panas MODIS, Indeks Iklim Global, dan Kekeringan Lokal

**Disusun oleh:** [Nama Anggota 1 – NIM], [Nama Anggota 2 – NIM], [Nama Anggota 3 – NIM]
**Mata Kuliah:** [Nama Mata Kuliah]
**Dosen Pengampu:** [Nama Dosen]
**Program Studi Teknik Informatika – [Nama Universitas]**
**[Bulan, Tahun]**

---

## BAB I PENDAHULUAN

### 1.1 Latar Belakang

Kebakaran hutan dan lahan (karhutla) terjadi hampir setiap tahun di Indonesia, terutama di Sumatra dan Kalimantan. Dampaknya mencakup kerusakan ekosistem, pelepasan emisi karbon, gangguan kesehatan akibat kabut asap, serta kerugian ekonomi dan gangguan aktivitas masyarakat yang dirasakan hingga negara tetangga. Musim kebakaran 2015 tercatat sebagai salah satu yang terparah sejak pengamatan satelit NASA dimulai pada awal 2000-an (Field dkk., 2016).

Penanganan karhutla selama ini cenderung reaktif: pemadaman dilakukan setelah titik api muncul. Sementara itu, personel, peralatan, dan anggaran terbatas, sedangkan wilayah rawan sangat luas. Dibutuhkan informasi yang menunjukkan **wilayah mana yang paling berisiko terbakar sebelum musim kebakaran tiba**, sehingga pencegahan dapat diprioritaskan.

Kejadian karhutla memiliki pola waktu dan tempat yang kuat. Analisis awal data titik panas MODIS tahun 2025 menunjukkan bahwa dari 17.847 titik panas yang terdeteksi di Indonesia, 14.162 titik merupakan dugaan kebakaran vegetasi dengan tingkat kepercayaan memadai. Jumlahnya sangat musiman, yaitu sekitar 101 titik pada Januari dan melonjak menjadi 3.845 titik pada Juli, serta terkonsentrasi di Kalimantan dan Sumatra. Penelitian sebelumnya juga menunjukkan bahwa aktivitas kebakaran dipengaruhi kekeringan yang berkaitan dengan El Niño dan fase Indian Ocean Dipole (Field dkk., 2016; Spessa dkk., 2015; Pan dkk., 2018). Pola musiman, spasial, dan iklim ini menunjukkan bahwa risiko karhutla berpotensi diprediksi.

Terdapat satu keterbatasan penting. Indeks El Niño seperti ONI bernilai **sama untuk seluruh Indonesia** pada satu bulan, sehingga tidak dapat membedakan wilayah mana yang kering. Kekeringan yang memicu kebakaran terjadi pada skala lokal, sehingga indeks global perlu dilengkapi data hujan, suhu, dan kelembapan per wilayah.

Sejumlah penelitian telah menerapkan machine learning dan deep learning untuk prediksi karhutla di Indonesia (Kadir dkk., 2023; Firmansyah dkk., 2024; Nikonovas dkk., 2022). Sebagian berfokus pada peramalan jumlah titik panas secara agregat, sedangkan sistem yang lebih kompleks bergantung pada prakiraan iklim musiman yang tidak mudah diakses. Penelitian ini mengusulkan model klasifikasi yang memprediksi **wilayah (grid) berisiko tinggi satu bulan ke depan** menggunakan data terbuka, yaitu titik panas MODIS, indeks iklim global (ONI dan DMI), serta hujan, suhu, dan kelembapan lokal dari NASA POWER. Hasilnya disajikan dalam dashboard peringatan dini berbasis web.

### 1.2 Rumusan Masalah

1. Bagaimana membangun model machine learning untuk memprediksi grid yang berisiko tinggi karhutla pada bulan berikutnya berdasarkan data titik panas MODIS, indeks iklim global, dan kekeringan lokal?
2. Algoritma mana yang memberikan kinerja terbaik di antara baseline persistensi, Logistic Regression, Random Forest, dan XGBoost?
3. Seberapa besar kontribusi setiap kelompok fitur (riwayat, musiman, spasial, ENSO, dan hujan lokal) terhadap kinerja prediksi, dan bagaimana kinerja model pada tahun El Niño?
4. Bagaimana menyajikan hasil prediksi dalam dashboard yang mudah dipahami sebagai alat bantu peringatan dini?

### 1.3 Tujuan Penelitian

1. Membangun model klasifikasi untuk memprediksi wilayah berisiko tinggi karhutla satu bulan ke depan.
2. Membandingkan kinerja beberapa algoritma dan menentukan algoritma terbaik berdasarkan metrik yang sesuai untuk data tidak seimbang.
3. Menganalisis kontribusi kelompok fitur melalui ablasi dan kepentingan fitur, serta menguji model pada tahun El Niño (2015, 2019, 2023).
4. Mengembangkan dashboard berbasis web yang menampilkan peta prediksi risiko karhutla per wilayah.

### 1.4 Manfaat Penelitian

1. **Bagi instansi penanggulangan bencana** (BPBD, Manggala Agni): informasi awal untuk memprioritaskan patroli dan pencegahan.
2. **Bagi masyarakat**: meningkatkan kewaspadaan terhadap potensi karhutla dan kabut asap.
3. **Bagi bidang keilmuan**: contoh alur sains data lengkap, dari pengolahan data satelit, rekayasa fitur, hingga pemodelan dan penyajian hasil.

### 1.5 Batasan Masalah

1. Wilayah penelitian adalah daratan Indonesia yang dibagi menjadi grid 0,5° × 0,5° (sekitar 55 × 55 km). Unit prediksi adalah sel yang pernah memiliki titik panas pada periode latih.
2. Data titik panas adalah MODIS Collection 6.1 dari NASA FIRMS periode 2012–2025.
3. Titik panas dipakai sebagai indikator kejadian kebakaran; tidak semua titik panas merupakan kebakaran yang terkonfirmasi di lapangan.
4. Variabel iklim terdiri atas ONI (NOAA CPC), DMI (NOAA PSL), serta hujan, suhu, dan kelembapan bulanan (NASA POWER).
5. Prediksi dilakukan untuk jangka satu bulan ke depan.
6. Model bersifat prediktif dan tidak dimaksudkan untuk menyimpulkan hubungan sebab-akibat.
7. Peta lahan gambut, tutupan lahan, dan batas administratif tidak termasuk cakupan awal dan menjadi pengembangan lanjutan.

---

## BAB II TINJAUAN PUSTAKA

### 2.1 Kebakaran Hutan dan Lahan serta Titik Panas

Titik panas (hotspot) adalah lokasi yang terdeteksi memiliki suhu lebih tinggi dibanding sekitarnya oleh sensor satelit. Data titik panas MODIS dihasilkan dari satelit Terra dan Aqua dengan algoritma deteksi api aktif Collection 6 (Giglio dkk., 2016) dan didistribusikan oleh NASA FIRMS. Setiap titik dilengkapi koordinat, waktu deteksi, tingkat kepercayaan (0–100), kekuatan radiasi api (FRP), dan tipe titik panas yang membedakan dugaan kebakaran vegetasi dari sumber panas lain seperti gunung api aktif dan sumber panas statis. Penyaringan berdasarkan tipe dan tingkat kepercayaan diperlukan agar data merepresentasikan kebakaran vegetasi.

### 2.2 El Niño, Indian Ocean Dipole, dan Kekeringan

El Niño membuat musim kemarau di Indonesia lebih kering dan panjang. Field dkk. (2016) menemukan bahwa aktivitas kebakaran memiliki sensitivitas nonlinear terhadap kekeringan, terutama ketika curah hujan berada di bawah sekitar 4 mm per hari dalam periode panjang, dan merekomendasikan peringatan dini yang dikaitkan dengan prakiraan curah hujan. Spessa dkk. (2015) menunjukkan bahwa aktivitas kebakaran berkorelasi negatif dengan curah hujan dan bahwa kebakaran besar dapat diprediksi beberapa bulan sebelumnya. Pan dkk. (2018) menambahkan bahwa jenis El Niño dan fase Indian Ocean Dipole turut memengaruhi kekeringan dan kebakaran.

Oceanic Niño Index (ONI) adalah indikator utama NOAA untuk El Niño, dihitung sebagai rata-rata bergerak tiga bulan anomali suhu permukaan laut di wilayah Niño 3.4. ONI ≥ +0,5 menandakan El Niño dan ≤ −0,5 menandakan La Niña. Karena berupa rata-rata tiga bulan, nilai ONI yang berpusat di bulan *m* baru diketahui setelah bulan *m+1* berakhir, sehingga pada penelitian ini ONI digunakan dengan lag satu bulan.

### 2.3 Algoritma Machine Learning

1. **Logistic Regression**: model klasifikasi linear yang memodelkan probabilitas kelas dengan fungsi sigmoid. Dipakai sebagai pembanding karena sederhana dan mudah diinterpretasikan.
2. **Random Forest**: ensemble banyak pohon keputusan yang dilatih pada sampel data dan subset fitur acak, dengan keputusan berdasarkan suara terbanyak (Breiman, 2001). Mampu menangkap hubungan nonlinear dan relatif tahan overfitting.
3. **XGBoost**: gradient boosting yang membangun pohon secara bertahap, setiap pohon baru memperbaiki kesalahan pohon sebelumnya, dengan regularisasi untuk mencegah overfitting (Chen & Guestrin, 2016).

Model berbasis pohon dipilih karena hubungan kekeringan dan kebakaran bersifat nonlinear (Field dkk., 2016).

### 2.4 Evaluasi pada Data Tidak Seimbang

Jumlah wilayah-bulan berisiko tinggi jauh lebih sedikit dibanding yang aman. Akurasi dapat menyesatkan karena model yang selalu menebak "aman" tetap tampak akurat. Evaluasi karena itu memakai precision, recall, F1-score, dan PR-AUC. Recall diutamakan karena dalam konteks bencana, kegagalan mendeteksi wilayah berisiko (false negative) lebih merugikan daripada peringatan yang tidak terjadi (false positive). Sari (2024) menunjukkan bahwa penanganan ketidakseimbangan kelas, misalnya pembobotan kelas pada Random Forest, memengaruhi kinerja prediksi titik panas.

### 2.5 Penelitian Terkait

| No | Peneliti (Tahun) | Data | Metode | Hasil / Temuan Utama |
|---|---|---|---|---|
| 1 | Field dkk. (2016) | Titik panas MODIS, curah hujan, data satelit asap | Analisis statistik | Kebakaran sangat sensitif secara nonlinear terhadap kekeringan akibat El Niño; merekomendasikan peringatan dini |
| 2 | Spessa dkk. (2015) | Area terbakar, curah hujan, tutupan pohon, prakiraan musiman ECMWF | Prakiraan musiman | Kebakaran di Kalimantan dapat diprediksi beberapa bulan sebelumnya |
| 3 | Pan dkk. (2018) | Kebakaran, kekeringan, indeks El Niño dan IOD, 1979–2016 | Analisis statistik | Jenis El Niño dan fase IOD memengaruhi kekeringan dan kebakaran |
| 4 | Nikonovas dkk. (2022) | Prakiraan musiman ECMWF SEAS5, tutupan hutan, gambut, titik panas MODIS | Multilayer Perceptron (ProbFire) | Peringatan dini probabilistik bulanan pada grid 0,25°; lebih baik dari klimatologi di sebagian wilayah |
| 5 | Yang dkk. (2021) | Citra Landsat 7, titik panas FIRMS | Neural network | Prediksi kebakaran satu bulan ke depan dengan AUC di atas 0,81 |
| 6 | Kadir dkk. (2023) | > 700.000 titik panas MODIS 2010–2021 | LSTM | Peramalan jumlah titik panas Indonesia tahun 2022 |
| 7 | Rosa dkk. (2022) | Titik panas MODIS Indonesia | LSTM | Pemetaan dan peramalan titik panas; konsentrasi tertinggi di Sumatra dan Kalimantan |
| 8 | Firmansyah dkk. (2024) | Titik panas dan data iklim (4.343 data) | Random Forest | Klasifikasi kasus kebakaran dengan kinerja baik |
| 9 | Sari (2024) | Titik panas Bangka Belitung 2023 | Neural Network, Random Forest | Random Forest dengan pembobotan kelas lebih baik pada data tidak seimbang |
| 10 | Santoso dkk. (2024) | Data patroli karhutla | Machine learning | Model prediksi potensi karhutla berbasis data patroli |

### 2.6 Posisi Penelitian

1. **Unit prediksi berbasis wilayah.** Sebagian penelitian (Kadir dkk., 2023; Rosa dkk., 2022) meramalkan jumlah titik panas agregat. Penelitian ini memprediksi status risiko per grid sehingga langsung menunjukkan wilayah prioritas.
2. **Data terbuka tanpa prakiraan musiman.** ProbFire (Nikonovas dkk., 2022) bergantung pada prakiraan iklim musiman yang kompleks. Penelitian ini memakai data publik yang mudah diperoleh (FIRMS, NOAA, NASA POWER) sehingga mudah direplikasi.
3. **Kekeringan lokal sebagai pelengkap indeks global.** ONI dan DMI dilengkapi hujan, suhu, dan kelembapan per grid agar model dapat membedakan wilayah.
4. **Rekayasa fitur yang dapat diuji.** Kontribusi tiap kelompok fitur diukur melalui ablasi dan permutation importance.
5. **Validasi pada tahun El Niño.** Selain data uji kronologis, model diuji dengan leave-one-year-out pada 2015, 2019, dan 2023.
6. **Luaran berupa aplikasi.** Hasil prediksi disajikan dalam dashboard interaktif.

---

## BAB III METODOLOGI PENELITIAN

### 3.1 Alur Penelitian

1. Pengumpulan data
2. EDA data mentah dan cleaning
3. Agregasi titik panas ke grid × bulan
4. Pengambilan data iklim lokal per grid
5. Rekayasa fitur dan pembuatan label
6. Pembagian data berdasarkan waktu
7. Pelatihan dan penyetelan model
8. Evaluasi, ablasi fitur, dan validasi tahun El Niño
9. Pengembangan dashboard

Seluruh tahap 1–8 diimplementasikan dalam satu notebook Python yang dapat dijalankan ulang (`karhutla_eda_pemodelan.ipynb`).

### 3.2 Sumber Data

| Data | Sumber | Periode | Peran |
|---|---|---|---|
| Titik panas MODIS Collection 6.1 (arsip per negara, CSV tahunan) | NASA FIRMS: https://firms.modaps.eosdis.nasa.gov/country/ | 2012–2025 | Label dan fitur riwayat |
| Oceanic Niño Index (ONI) | NOAA CPC: https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt | 2012–2025 | Fase El Niño / La Niña |
| Dipole Mode Index (DMI) | NOAA PSL: https://psl.noaa.gov/gcos_wgsp/Timeseries/Data/dmi.had.long.data | 2012–2025 | Fase Indian Ocean Dipole |
| Curah hujan (PRECTOTCORR), suhu 2 m (T2M), kelembapan relatif (RH2M) bulanan | NASA POWER API: https://power.larc.nasa.gov/ | 2012–2025 | Kekeringan lokal per grid |

### 3.3 EDA dan Cleaning

EDA dilakukan sebelum pembersihan untuk memeriksa struktur, nilai kosong, duplikat, serta sebaran tipe, kepercayaan, FRP, dan koordinat. Aturan pembersihan yang diterapkan berurutan, dengan jumlah baris yang dibuang dicatat pada setiap langkah:

1. Membuang baris dengan tanggal, koordinat, atau kepercayaan tidak valid.
2. Membuang duplikat (koordinat, tanggal, jam, dan satelit yang sama).
3. Membuang koordinat di luar batas Indonesia.
4. Mempertahankan hanya titik panas tipe 0 (dugaan kebakaran vegetasi), membuang gunung api aktif, sumber panas statis, dan titik lepas pantai.
5. Mempertahankan hanya titik panas dengan tingkat kepercayaan ≥ 30.

Nilai FRP yang besar dipertahankan karena mencerminkan kebakaran besar yang nyata, bukan galat pengukuran.

EDA lanjutan mencakup pola musiman (heatmap tahun × bulan), hubungan ENSO dengan total titik panas tahunan, perbandingan sebaran spasial saat El Niño dan non-El Niño, hubungan curah hujan dengan titik panas, serta proporsi kelas positif pada beberapa kandidat ambang.

### 3.4 Agregasi, Rekayasa Fitur, dan Label

**Agregasi.** Titik panas dipetakan ke grid 0,5° dengan `floor(koordinat / ukuran grid)`, lalu dihitung per grid per bulan. Kombinasi grid-bulan tanpa titik panas diisi nol. Sel yang dipakai sebagai unit prediksi adalah sel yang pernah memiliki titik panas pada periode latih.

**Fitur (22 fitur dalam 5 kelompok).** Seluruh fitur hanya memakai informasi yang tersedia pada akhir bulan *t*.

| Kelompok | Fitur | Keterangan |
|---|---|---|
| Riwayat | `hotspot_bulan_ini`, `hotspot_bulan_lalu`, `hotspot_2bulan_lalu`, `rata2_3bulan`, `frp_rata2_ini` | Aktivitas api terkini |
| Musiman | `hotspot_tahun_lalu`, `rata2_historis_bulan`, `bulan_target`, `bulan_sin`, `bulan_cos` | Pola bulan yang diprediksi; rata-rata historis dihitung ekspansif sehingga hanya memuat tahun sebelumnya |
| Spasial | `hotspot_tetangga_ini`, `grid_lat`, `grid_lon` | Jumlah titik panas di 8 sel tetangga dan posisi grid |
| ENSO | `oni_lag1`, `oni_delta3`, `dmi_lag1` | ONI dan DMI dengan lag agar tidak memakai data masa depan |
| Hujan lokal | `hujan_bulan_ini`, `anomali_hujan_ini`, `anomali_hujan_3bulan`, `bulan_kering_beruntun`, `anomali_suhu_ini`, `rh_bulan_ini` | Kekeringan per grid; bulan kering ditandai curah hujan < 4 mm/hari (Field dkk., 2016); klimatologi hanya memakai tahun latih |

**Label.** `risiko_tinggi` = 1 jika jumlah titik panas grid pada bulan *t+1* mencapai ambang (nilai awal: ≥ 10 titik per grid per bulan, ditinjau ulang berdasarkan distribusi data pada EDA), dan 0 jika tidak.

**Pemeriksaan konsistensi.** Sampel baris dihitung ulang secara independen dari data sumber untuk memastikan fitur dan label sesuai definisi dan tidak memuat informasi setelah bulan *t*.

### 3.5 Data Latih dan Pembagian Data

Data dibagi berdasarkan **tahun bulan target**, bukan secara acak, agar model diuji pada periode yang belum pernah dilihat.

| Subset | Periode target | Kegunaan |
|---|---|---|
| Data latih | 2012–2022 | Melatih model; menentukan daftar sel dan klimatologi hujan |
| Data validasi | 2023–2024 | Memilih hyperparameter, model, dan ambang probabilitas (mencakup El Niño 2023) |
| Data uji | 2025 | Evaluasi akhir |
| Bulan depan | Setelah data terakhir | Prediksi untuk dashboard memakai model yang dilatih ulang pada seluruh data berlabel |

Satu baris data latih adalah satu pasangan **grid × bulan** dengan 22 fitur dan satu label. Jumlah baris, jumlah sel, dan proporsi kelas positif dilaporkan dari hasil eksekusi notebook.

### 3.6 Pemodelan

1. **Baseline persistensi**: memprediksi "berisiko tinggi" bulan depan jika bulan ini berisiko tinggi. Model lain harus mengungguli baseline ini.
2. **Logistic Regression** dengan imputasi median, standardisasi fitur, dan pembobotan kelas.
3. **Random Forest** dengan `class_weight="balanced"`.
4. **XGBoost** dengan `scale_pos_weight` sebesar rasio kelas negatif terhadap positif pada data latih.

Hyperparameter dipilih dengan grid kecil berdasarkan **PR-AUC pada data validasi** (Logistic Regression: C; Random Forest: kedalaman dan minimum sampel daun; XGBoost: jumlah pohon, kedalaman, dan learning rate). Implementasi memakai Python dengan pandas, scikit-learn, dan xgboost.

### 3.7 Evaluasi

1. **Pemilihan ambang probabilitas** pada data validasi: F1 tertinggi dengan syarat recall minimal 0,70.
2. **Perbandingan model** dengan precision, recall, F1-score, PR-AUC, ROC-AUC, dan confusion matrix. Model terbaik dipilih berdasarkan F1 validasi dan dievaluasi pada data uji 2025.
3. **Kepentingan fitur** dengan permutation importance pada data validasi.
4. **Kinerja per fase ENSO** pada gabungan data validasi dan uji.
5. **Leave-one-year-out** untuk tahun El Niño 2015, 2019, dan 2023: satu tahun target ditahan, model dilatih ulang pada tahun lainnya. Skema ini mengukur generalisasi lintas tahun El Niño, bukan urutan kronologis.
6. **Ablasi kelompok fitur**: setiap kelompok dilepas satu per satu untuk melihat perubahan PR-AUC.

### 3.8 Pengembangan Dashboard

Dashboard dikembangkan dengan Streamlit dan hanya membaca berkas hasil notebook (tanpa mengunduh data atau melatih model):

1. Pilihan bulan prediksi dan pulau.
2. Kartu ringkasan jumlah grid berisiko tinggi.
3. Peta interaktif grid berwarna menurut probabilitas risiko.
4. Pilihan skenario fase ENSO (El Niño, Netral, La Niña) berupa frekuensi historis kejadian berisiko tinggi per grid dan bulan.
5. Tabel wilayah prioritas dengan probabilitas tertinggi.
6. Halaman evaluasi: perbandingan model, kepentingan fitur, ablasi, dan validasi tahun El Niño.

Skenario ENSO disusun dari frekuensi historis, bukan dengan mengubah nilai ONI pada model, karena hujan lokal bergantung pada ONI sehingga mengubah ONI saja menghasilkan kombinasi masukan yang tidak pernah terjadi.

---

## BAB IV RENCANA PELAKSANAAN

### 4.1 Jadwal Kegiatan

| No | Kegiatan | M1 | M2 | M3 | M4 | M5 | M6 | M7 | M8 |
|---|---|---|---|---|---|---|---|---|---|
| 1 | Studi literatur dan finalisasi proposal | ■ | | | | | | | |
| 2 | Pengumpulan data | ■ | ■ | | | | | | |
| 3 | EDA, cleaning, dan agregasi | | ■ | ■ | | | | | |
| 4 | Rekayasa fitur | | | ■ | ■ | | | | |
| 5 | Pemodelan dan penyetelan | | | | ■ | ■ | | | |
| 6 | Evaluasi, ablasi, dan validasi El Niño | | | | | ■ | ■ | | |
| 7 | Pengembangan dashboard | | | | | | ■ | ■ | |
| 8 | Laporan dan presentasi | | | | | | | ■ | ■ |

*(Sesuaikan jumlah minggu dengan tenggat tugas.)*

### 4.2 Pembagian Tugas

| Anggota | Tanggung Jawab Utama |
|---|---|
| [Anggota 1] | Pengumpulan data, EDA, cleaning, dan rekayasa fitur |
| [Anggota 2] | Pemodelan, penyetelan, evaluasi, dan ablasi |
| [Anggota 3] | Dashboard dan visualisasi |
| Semua | Studi literatur, laporan, dan presentasi |

### 4.3 Luaran yang Diharapkan

1. Dataset grid × bulan hasil pengolahan titik panas, indeks iklim, dan data iklim lokal.
2. Notebook analisis dan pemodelan yang dapat dijalankan ulang.
3. Model prediksi terbaik beserta perbandingan algoritma, ablasi fitur, dan validasi tahun El Niño.
4. Dashboard peringatan dini berbasis web.
5. Laporan akhir dan kode sumber (repositori GitHub).

### 4.4 Risiko dan Mitigasi

| Risiko | Mitigasi |
|---|---|
| Hanya ada sedikit episode El Niño pada data latih | Validasi leave-one-year-out pada 2015, 2019, dan 2023; kinerja dilaporkan per fase ENSO |
| Titik panas tertutup awan atau asap sehingga terhitung kurang | Dicantumkan sebagai keterbatasan; hotspot diperlakukan sebagai indikator, bukan kebakaran terkonfirmasi |
| Unduhan FIRMS atau NASA POWER gagal | Cache lokal dan opsi unggah manual berkas ke folder `data_raw/` |
| Kelas positif sangat sedikit | Pembobotan kelas, metrik PR-AUC dan recall, serta pemilihan ambang pada data validasi |
| Kebocoran data antarwaktu | Pembagian berdasarkan waktu, lag ONI dan DMI, klimatologi dari tahun latih saja, dan pemeriksaan konsistensi fitur |

---

## DAFTAR PUSTAKA

Breiman, L. (2001). Random forests. *Machine Learning*, 45(1), 5–32.

Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, 785–794.

Field, R. D., van der Werf, G. R., Fanin, T., Fetzer, E. J., Fuller, R., Jethva, H., Levy, R., Livesey, N. J., Luo, M., Torres, O., & Worden, H. M. (2016). Indonesian fire activity and smoke pollution in 2015 show persistent nonlinear sensitivity to El Niño-induced drought. *Proceedings of the National Academy of Sciences*, 113(33), 9204–9209. https://doi.org/10.1073/pnas.1524888113

Firmansyah, A., Syahidin, M. F., & Triana, Y. S. (2024). Prediksi kebakaran hutan berdasarkan titik panas dan iklim menggunakan algoritma Random Forest. *Jurnal Nasional Teknologi dan Sistem Informasi (TEKNOSI)*, 10(2), 145–155.

Giglio, L., Schroeder, W., & Justice, C. O. (2016). The collection 6 MODIS active fire detection algorithm and fire products. *Remote Sensing of Environment*, 178, 31–41.

Kadir, E. A., Kung, H. T., Nasution, A. H., Daud, H., AlMansour, A. A., Othman, M., & Rosa, S. L. (2023). Fires hotspot forecasting in Indonesia using Long Short-Term Memory algorithm and MODIS datasets. Dalam *Springer International Publishing* (hlm. 589–602).

Nikonovas, T., Spessa, A., Doerr, S. H., Clay, G. D., & Mezbahuddin, S. (2022). ProbFire: A probabilistic fire early warning system for Indonesia. *Natural Hazards and Earth System Sciences*, 22(2), 303–322. https://doi.org/10.5194/nhess-22-303-2022

Pan, X., Chin, M., Ichoku, C. M., & Field, R. D. (2018). Connecting Indonesian fires and drought with the type of El Niño and phase of the Indian Ocean Dipole during 1979–2016. *Journal of Geophysical Research: Atmospheres*. https://doi.org/10.1029/2018JD028402

Rosa, S. L., dkk. (2022). Fire hotspots mapping and forecasting in Indonesia using deep learning algorithm. *Proceedings of the 3rd International Conference on Electrical Engineering and Informatics (ICon EEI 2022)*, 190–.

Santoso, A. B., Sitanggang, I. S., & Hardhienata, M. K. D. (2024). Pembangunan model prediksi potensi kebakaran hutan dan lahan menggunakan algoritma machine learning berdasarkan data patroli. *Jurnal INFOTEL*, 16(3).

Sari, D. N. (2024). *Prediksi tingkat bahaya kemunculan titik panas penyebab kebakaran hutan dan lahan menggunakan metode Neural Network dan Random Forest* [Skripsi]. Institut Teknologi Sepuluh Nopember.

Spessa, A. C., Field, R. D., Pappenberger, F., Langner, A., Englhart, S., Weber, U., Stockdale, T., Siegert, F., Kaiser, J. W., & Moore, J. (2015). Seasonal forecasting of fire over Kalimantan, Indonesia. *Natural Hazards and Earth System Sciences*, 15(3), 429–442. https://doi.org/10.5194/nhess-15-429-2015

Yang, S., Lupascu, M., & Meel, K. S. (2021). Predicting forest fire using remote sensing data and machine learning. *Proceedings of the AAAI Conference on Artificial Intelligence*, 35(17), 14983–14990.

**Sumber data:**

NASA FIRMS. *MODIS Collection 6.1 Active Fire Detections (MCD14ML)*. https://firms.modaps.eosdis.nasa.gov/

NOAA Climate Prediction Center. *Oceanic Niño Index (ONI)*. https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt

NOAA Physical Sciences Laboratory. *Dipole Mode Index (DMI)*. https://psl.noaa.gov/gcos_wgsp/Timeseries/Data/dmi.had.long.data

NASA POWER. *Prediction Of Worldwide Energy Resources*. https://power.larc.nasa.gov/
