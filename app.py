import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Dashboard Risiko Karhutla", page_icon="🔥", layout="wide")

BASE = Path(__file__).parent
DATA = Path(os.environ.get("KARHUTLA_DATA", BASE / "data"))
if not (DATA / "metrik.json").exists() and (BASE / "data_contoh" / "metrik.json").exists():
    DATA = BASE / "data_contoh"
BULAN = ["Januari", "Februari", "Maret", "April", "Mei", "Juni", "Juli",
         "Agustus", "September", "Oktober", "November", "Desember"]
FASE = ["El Niño", "Netral", "La Niña"]
WARNA_AKTUAL = {"Risiko tinggi": "#d9481f", "Tidak berisiko tinggi": "#e6dcd6"}
BERKAS_WAJIB = ["grid.geojson", "info_sel.csv", "prediksi_riwayat.csv", "skenario_enso.csv",
                "perbandingan_model.csv", "feature_importance.csv", "ablasi_fitur.csv",
                "validasi_el_nino.csv", "metrik.json"]


def label_periode(p):
    tahun, bulan = p.split("-")
    return f"{BULAN[int(bulan) - 1]} {tahun}"


@st.cache_data(show_spinner="Memuat data...")
def muat():
    d = {
        "geo": json.loads((DATA / "grid.geojson").read_text()),
        "info": pd.read_csv(DATA / "info_sel.csv"),
        "riwayat": pd.read_csv(DATA / "prediksi_riwayat.csv"),
        "skenario": pd.read_csv(DATA / "skenario_enso.csv"),
        "banding": pd.read_csv(DATA / "perbandingan_model.csv"),
        "penting": pd.read_csv(DATA / "feature_importance.csv"),
        "ablasi": pd.read_csv(DATA / "ablasi_fitur.csv"),
        "loyo": pd.read_csv(DATA / "validasi_el_nino.csv"),
        "metrik": json.loads((DATA / "metrik.json").read_text()),
    }
    depan = DATA / "prediksi_bulan_depan.csv"
    d["depan"] = pd.read_csv(depan) if depan.exists() else None
    d["riwayat"] = d["riwayat"].merge(d["info"], on="grid_id", how="left")
    d["skenario"] = d["skenario"].merge(d["info"], on="grid_id", how="left")
    return d


kurang = [b for b in BERKAS_WAJIB if not (DATA / b).exists()]
if kurang:
    st.error("Berkas hasil notebook belum lengkap.")
    st.markdown(
        f"Folder data yang dicari: `{DATA}`\n\nBerkas yang belum ada: " + ", ".join(f"`{b}`" for b in kurang)
        + "\n\nJalankan notebook `karhutla_eda_pemodelan.ipynb` sampai selesai, lalu salin isi folder "
        "`output_dashboard/` ke folder `data/` di samping `app.py`."
    )
    st.stop()

D = muat()
M = D["metrik"]
if M.get("data_tiruan"):
    st.warning("Menampilkan DATA CONTOH TIRUAN untuk mencoba aplikasi. Angka dan peta di sini bukan hasil nyata. "
               "Salin hasil notebook ke folder `data/` untuk menampilkan hasil sebenarnya.")
AMBANG = float(M["ambang_probabilitas"])


def tingkat(proba):
    return np.where(proba >= AMBANG, "Tinggi", np.where(proba >= AMBANG / 2, "Sedang", "Rendah"))


def peta(df, warna, label, skala="YlOrRd", rentang=(0, 1), diskrit=None, hover=None, tinggi=560, tengah_nol=False):
    if df.empty:
        st.info("Tidak ada grid yang memenuhi filter.")
        return
    lat, lon = df["lat"], df["lon"]
    span = max(lon.max() - lon.min(), (lat.max() - lat.min()) * 1.6, 2.0)
    zoom = float(np.clip(np.log2(360 * 800 / (256 * span * 2.2)), 3.0, 8.0))
    argumen = dict(
        geojson=D["geo"], locations="grid_id", featureidkey="properties.grid_id", color=warna,
        hover_name="grid_id", hover_data=hover, map_style=os.environ.get("KARHUTLA_MAP_STYLE", "carto-positron"), opacity=0.78, zoom=zoom,
        center={"lat": float((lat.max() + lat.min()) / 2), "lon": float((lon.max() + lon.min()) / 2)},
        labels=label,
    )
    if diskrit:
        argumen["color_discrete_map"] = diskrit
    else:
        argumen["color_continuous_scale"] = skala
        argumen["range_color"] = rentang
        if tengah_nol:
            argumen["color_continuous_midpoint"] = 0
    fig = px.choropleth_map(df, **argumen)
    fig.update_traces(marker_line_width=0.4, marker_line_color="rgba(255,255,255,0.7)")
    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0), height=tinggi, legend_title_text="",
        coloraxis_colorbar=dict(orientation="h", x=0.99, y=0.99, xanchor="right", yanchor="top",
                                len=0.4, thickness=10, title=dict(side="top")),
    )
    st.plotly_chart(fig, width="stretch")


def kolom_tabel(df, kolom):
    return df[list(kolom)].rename(columns=kolom)


def unduh(df, nama, kunci):
    st.download_button("Unduh tabel (CSV)", df.to_csv(index=False).encode("utf-8"), nama, "text/csv", key=kunci)


with st.sidebar:
    st.title("🔥 Risiko Karhutla")
    st.caption("Prediksi wilayah rawan kebakaran hutan dan lahan per grid 0,5°")
    halaman = st.radio("Halaman", ["Peta Risiko", "Skenario El Niño", "Riwayat vs Aktual", "Evaluasi Model", "Tentang"],
                       label_visibility="collapsed")
    st.divider()
    st.subheader("Filter")
    daftar_pulau = sorted(D["info"]["pulau"].dropna().unique())
    pulau_pilih = st.multiselect("Pulau", daftar_pulau, default=daftar_pulau, key="pulau")
    st.caption(f"Ambang peringatan model: **{AMBANG:.2f}**. Tinggi ≥ ambang, sedang ≥ setengah ambang.")


def pilih_pulau(df):
    return df[df["pulau"].isin(pulau_pilih)] if pulau_pilih else df.iloc[0:0]


def halaman_peta():
    st.header("Peta Risiko")
    opsi = (["Prediksi bulan depan"] if D["depan"] is not None else []) + ["Riwayat prediksi (validasi dan uji)"]
    k1, k2, k3 = st.columns([1.3, 1.3, 1])
    sumber = k1.radio("Sumber", opsi, horizontal=False)

    if sumber == "Prediksi bulan depan":
        df = D["depan"].copy()
        periode = df["bulan_prediksi"].iloc[0]
        k2.metric("Bulan yang diprediksi", label_periode(periode))
        kolom = {"grid_id": "Grid", "pulau": "Pulau", "lat": "Lintang", "lon": "Bujur", "proba": "Probabilitas",
                 "tingkat": "Tingkat", "hotspot_bulan_ini": "Titik panas bulan ini"}
    else:
        daftar = sorted(D["riwayat"]["bulan_prediksi"].unique())
        periode = k2.select_slider("Bulan yang diprediksi", daftar, value=daftar[-1], format_func=label_periode)
        df = D["riwayat"][D["riwayat"]["bulan_prediksi"] == periode].copy()
        kolom = {"grid_id": "Grid", "pulau": "Pulau", "lat": "Lintang", "lon": "Bujur", "proba": "Probabilitas",
                 "tingkat": "Tingkat", "hotspot_aktual": "Titik panas aktual"}
    hanya = k3.toggle("Hanya di atas ambang", value=False)

    df = pilih_pulau(df)
    df["tingkat"] = tingkat(df["proba"])
    tampil = df[df["proba"] >= AMBANG] if hanya else df

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Grid dianalisis", f"{len(df):,}".replace(",", "."))
    m2.metric("Risiko tinggi", int((df["tingkat"] == "Tinggi").sum()))
    m3.metric("Risiko sedang", int((df["tingkat"] == "Sedang").sum()))
    if len(df):
        atas = df.loc[df["proba"].idxmax()]
        m4.metric("Probabilitas tertinggi", f"{atas['proba']:.2f}", f"{atas['pulau']} · {atas['grid_id']}", delta_color="off")
    else:
        m4.metric("Probabilitas tertinggi", "-")

    peta(tampil, "proba", {"proba": "Probabilitas"}, hover={"pulau": True, "tingkat": True, "lat": ":.2f", "lon": ":.2f"})
    if sumber == "Prediksi bulan depan" and len(df) and (df["tingkat"] == "Tinggi").sum() == 0:
        st.info("Tidak ada grid di atas ambang pada bulan ini. Bulan prediksi berada di luar musim kebakaran, "
                "sehingga hasil ini normal.")

    st.subheader("Wilayah prioritas")
    n = st.slider("Jumlah grid ditampilkan", 5, 50, 15, step=5)
    tabel = kolom_tabel(df.sort_values("proba", ascending=False).head(n), kolom)
    st.dataframe(tabel, width="stretch", hide_index=True, column_config={
        "Probabilitas": st.column_config.ProgressColumn(format="%.2f", min_value=0, max_value=1),
        "Lintang": st.column_config.NumberColumn(format="%.2f"),
        "Bujur": st.column_config.NumberColumn(format="%.2f"),
    })
    unduh(tabel, f"prioritas_{periode}.csv", "unduh_peta")
    st.caption("Titik panas adalah indikator, bukan kebakaran yang terkonfirmasi di lapangan.")


def halaman_skenario():
    st.header("Skenario El Niño")
    st.write("Frekuensi historis kejadian risiko tinggi pada sebuah grid dan bulan, dipisah menurut fase ENSO yang "
             "tercatat sebulan sebelumnya. Ini data historis per fase, bukan hasil mengubah nilai ONI pada model.")
    c1, c2, c3 = st.columns(3)
    bulan = c1.select_slider("Bulan target", list(range(1, 13)), value=9, format_func=lambda m: BULAN[m - 1])
    mode = c2.radio("Tampilan", ["Satu fase", "Selisih dua fase"], horizontal=True)
    metrik = c3.radio("Ukuran", ["Frekuensi risiko tinggi", "Rata-rata titik panas"], horizontal=True)
    kolom_ukur = "frekuensi_risiko_tinggi" if metrik == "Frekuensi risiko tinggi" else "rata2_hotspot"

    n_min = st.slider("Minimal bulan pengamatan per fase", 1, 6, 2,
                      help="Fase dengan pengamatan sedikit (terutama El Niño dan La Niña) kurang andal.")
    dasar = pilih_pulau(D["skenario"][D["skenario"]["bulan_target_ke"] == bulan])
    dasar = dasar[dasar["n_bulan"] >= n_min]

    if mode == "Satu fase":
        fase = st.radio("Fase ENSO", FASE, horizontal=True)
        df = dasar[dasar["fase_enso"] == fase]
        rentang = (0, 1) if kolom_ukur == "frekuensi_risiko_tinggi" else (0, max(float(df[kolom_ukur].max()), 1.0) if len(df) else 1)
        peta(df, kolom_ukur, {kolom_ukur: metrik}, rentang=rentang,
             hover={"pulau": True, "n_bulan": True, "lat": False, "lon": False})
        st.caption(f"{len(df)} grid memiliki minimal {n_min} bulan pengamatan pada fase {fase}.")
    else:
        a, b = st.columns(2)
        fa = a.selectbox("Fase A", FASE, index=0)
        fb = b.selectbox("Fase B", FASE, index=1)
        if fa == fb:
            st.warning("Pilih dua fase yang berbeda.")
            return
        pa = dasar[dasar["fase_enso"] == fa].set_index("grid_id")[kolom_ukur]
        pb = dasar[dasar["fase_enso"] == fb].set_index("grid_id")[kolom_ukur]
        selisih = (pa - pb).dropna().rename("selisih").reset_index().merge(D["info"], on="grid_id")
        selisih = pilih_pulau(selisih)
        batas = max(float(selisih["selisih"].abs().max()), 0.05) if len(selisih) else 1
        peta(selisih, "selisih", {"selisih": f"{fa} − {fb}"}, skala="RdBu_r", rentang=(-batas, batas),
             tengah_nol=True, hover={"pulau": True, "lat": False, "lon": False})
        st.caption(f"Merah: {fa} lebih tinggi. Biru: {fb} lebih tinggi. Hanya grid yang punya data pada kedua fase.")
        dasar = dasar[dasar["fase_enso"].isin([fa, fb])]

    ringkas = dasar.groupby(["pulau", "fase_enso"])[kolom_ukur].mean().reset_index()
    if len(ringkas):
        st.subheader("Ringkasan per pulau")
        fig = px.bar(ringkas, x="pulau", y=kolom_ukur, color="fase_enso", barmode="group",
                     category_orders={"fase_enso": FASE},
                     color_discrete_map={"El Niño": "#d9481f", "Netral": "#a6a6a6", "La Niña": "#2a7f8e"},
                     labels={kolom_ukur: metrik, "pulau": "", "fase_enso": "Fase"})
        fig.update_layout(height=340, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig, width="stretch")


def halaman_riwayat():
    st.header("Riwayat vs Aktual")
    st.write("Membandingkan prediksi model dengan kejadian sebenarnya pada periode validasi dan uji.")
    r = D["riwayat"].dropna(subset=["aktual"]).copy()
    r["tingkat"] = tingkat(r["proba"])
    r["prediksi_tinggi"] = (r["proba"] >= AMBANG).astype(int)
    r = pilih_pulau(r)
    if r.empty:
        st.info("Tidak ada data untuk filter ini.")
        return
    daftar = sorted(r["bulan_prediksi"].unique())
    periode = st.select_slider("Bulan", daftar, value=daftar[-1], format_func=label_periode)
    s = r[r["bulan_prediksi"] == periode].copy()
    s["aktual_label"] = np.where(s["aktual"] == 1, "Risiko tinggi", "Tidak berisiko tinggi")

    tp = int(((s["prediksi_tinggi"] == 1) & (s["aktual"] == 1)).sum())
    fp = int(((s["prediksi_tinggi"] == 1) & (s["aktual"] == 0)).sum())
    fn = int(((s["prediksi_tinggi"] == 0) & (s["aktual"] == 1)).sum())
    tn = int(((s["prediksi_tinggi"] == 0) & (s["aktual"] == 0)).sum())
    presisi = tp / (tp + fp) if tp + fp else np.nan
    recall = tp / (tp + fn) if tp + fn else np.nan
    m = st.columns(6)
    m[0].metric("Terdeteksi benar", tp)
    m[1].metric("Peringatan keliru", fp)
    m[2].metric("Terlewat", fn)
    m[3].metric("Aman benar", tn)
    m[4].metric("Precision", "-" if np.isnan(presisi) else f"{presisi:.2f}")
    m[5].metric("Recall", "-" if np.isnan(recall) else f"{recall:.2f}")

    a, b = st.columns(2)
    with a:
        st.subheader("Probabilitas prediksi")
        peta(s, "proba", {"proba": "Probabilitas"}, tinggi=460, hover={"pulau": True, "lat": False, "lon": False})
    with b:
        st.subheader("Kejadian aktual")
        peta(s, "aktual_label", {"aktual_label": "Status"}, diskrit=WARNA_AKTUAL, tinggi=460,
             hover={"pulau": True, "hotspot_aktual": True, "lat": False, "lon": False})

    terlewat = s[(s["prediksi_tinggi"] == 0) & (s["aktual"] == 1)].sort_values("proba")
    if len(terlewat):
        st.subheader("Grid yang terlewat")
        st.dataframe(kolom_tabel(terlewat.head(20), {"grid_id": "Grid", "pulau": "Pulau", "proba": "Probabilitas",
                                                      "hotspot_aktual": "Titik panas aktual"}),
                     width="stretch", hide_index=True)

    st.subheader("Jumlah grid berisiko tinggi per bulan")
    waktu = r.groupby("bulan_prediksi").agg(Prediksi=("prediksi_tinggi", "sum"), Aktual=("aktual", "sum")).reset_index()
    waktu = waktu.melt("bulan_prediksi", var_name="Seri", value_name="Jumlah grid")
    fig = px.line(waktu, x="bulan_prediksi", y="Jumlah grid", color="Seri", markers=True,
                  color_discrete_map={"Prediksi": "#2a7f8e", "Aktual": "#d9481f"}, labels={"bulan_prediksi": ""})
    fig.add_vline(x=periode, line_dash="dash", line_color="#888")
    fig.update_layout(height=320, margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig, width="stretch")


def halaman_evaluasi():
    st.header("Evaluasi Model")
    st.write(f"Model terbaik: **{M['model_terbaik']}** (dipilih dari F1 pada data validasi).")
    t1, t2, t3, t4 = st.tabs(["Perbandingan model", "Kepentingan fitur", "Ablasi fitur", "Tahun El Niño"])

    with t1:
        b = D["banding"].copy()
        panjang = b.melt(id_vars="model", value_vars=["val_f1", "uji_f1", "val_pr_auc", "uji_pr_auc"])
        panjang["metrik"] = panjang["variable"].map({"val_f1": "F1 validasi", "uji_f1": "F1 uji",
                                                      "val_pr_auc": "PR-AUC validasi", "uji_pr_auc": "PR-AUC uji"})
        pilih = st.radio("Metrik", ["F1", "PR-AUC"], horizontal=True)
        sub = panjang[panjang["metrik"].str.startswith(pilih)]
        fig = px.bar(sub, x="model", y="value", color="metrik", barmode="group", labels={"value": pilih, "model": ""})
        fig.update_layout(height=360, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig, width="stretch")
        tampil = b[["model", "ambang", "val_precision", "val_recall", "val_f1", "uji_precision", "uji_recall", "uji_f1",
                    "uji_pr_auc"]].rename(columns={
            "model": "Model", "ambang": "Ambang", "val_precision": "Precision val", "val_recall": "Recall val",
            "val_f1": "F1 val", "uji_precision": "Precision uji", "uji_recall": "Recall uji", "uji_f1": "F1 uji",
            "uji_pr_auc": "PR-AUC uji"})
        st.dataframe(tampil, width="stretch", hide_index=True, column_config={
            c: st.column_config.NumberColumn(format="%.3f") for c in tampil.columns if c != "Model"})
        u = M["uji"]
        cm = np.array([[u["tn"], u["fp"]], [u["fn"], u["tp"]]])
        fig = px.imshow(cm, text_auto=True, x=["Aman", "Berisiko"], y=["Aman", "Berisiko"],
                        color_continuous_scale="Blues", labels=dict(x="Prediksi", y="Aktual", color="Jumlah"))
        fig.update_layout(height=340, margin=dict(l=0, r=0, t=30, b=0), title="Confusion matrix data uji",
                          coloraxis_showscale=False)
        st.plotly_chart(fig, width="stretch")

    with t2:
        p = D["penting"].sort_values("penurunan_pr_auc")
        fig = px.bar(p, x="penurunan_pr_auc", y="fitur", color="kelompok", orientation="h",
                     labels={"penurunan_pr_auc": "Penurunan PR-AUC saat fitur diacak", "fitur": ""})
        fig.update_layout(height=620, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig, width="stretch")
        st.caption("Permutation importance pada data validasi. Fitur berkorelasi dapat saling menutupi kontribusinya.")

    with t3:
        a = D["ablasi"].copy()
        panjang = a.melt(id_vars="konfigurasi", value_vars=["pr_auc_validasi", "pr_auc_uji"])
        panjang["data"] = panjang["variable"].map({"pr_auc_validasi": "Validasi", "pr_auc_uji": "Uji"})
        fig = px.bar(panjang, x="konfigurasi", y="value", color="data", barmode="group",
                     labels={"value": "PR-AUC", "konfigurasi": ""})
        fig.update_layout(height=380, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig, width="stretch")
        st.dataframe(a.rename(columns={"konfigurasi": "Konfigurasi", "pr_auc_validasi": "PR-AUC val",
                                       "pr_auc_uji": "PR-AUC uji", "selisih_validasi": "Selisih val",
                                       "selisih_uji": "Selisih uji"}),
                     width="stretch", hide_index=True)
        st.caption("Setiap kelompok fitur dilepas satu per satu. Penurunan besar berarti kelompok itu penting.")

    with t4:
        l = D["loyo"].copy()
        panjang = l.melt(id_vars="tahun_ditahan", value_vars=["pr_auc_model", "pr_auc_persistensi"])
        panjang["model"] = panjang["variable"].map({"pr_auc_model": M["model_terbaik"], "pr_auc_persistensi": "Persistensi"})
        panjang["tahun_ditahan"] = panjang["tahun_ditahan"].astype(str)
        fig = px.bar(panjang, x="tahun_ditahan", y="value", color="model", barmode="group",
                     labels={"value": "PR-AUC", "tahun_ditahan": "Tahun ditahan"})
        fig.update_layout(height=360, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig, width="stretch")
        st.dataframe(l, width="stretch", hide_index=True)
        st.caption("Leave-one-year-out: satu tahun target ditahan, model dilatih pada tahun lainnya.")


def halaman_tentang():
    st.header("Tentang")
    st.markdown(
        "Dashboard ini menampilkan hasil model yang memprediksi **grid 0,5° yang berisiko tinggi karhutla** "
        "pada bulan berikutnya. Seluruh pengolahan dan pelatihan dilakukan di notebook; aplikasi ini hanya "
        "membaca berkas hasilnya."
    )
    c1, c2, c3 = st.columns(3)
    c1.metric("Model", M["model_terbaik"])
    c2.metric("Jumlah grid", M["jumlah_sel"])
    c3.metric("Ambang risiko tinggi", f"≥ {M['ambang_risiko_hotspot']} titik panas")
    st.markdown(
        f"**Periode data:** {M['periode_data'][0]}–{M['periode_data'][1]}  \n"
        f"**Pembagian waktu:** latih sampai {M['split']['latih_sampai']}, validasi "
        f"{M['split']['validasi'][0]}–{M['split']['validasi'][1]}, uji {M['split']['uji']}  \n"
        f"**Ambang probabilitas:** {AMBANG:.3f}"
    )
    st.subheader("Sumber data")
    st.markdown("- Titik panas MODIS Collection 6.1, NASA FIRMS\n- Oceanic Niño Index, NOAA CPC\n"
                "- Dipole Mode Index, NOAA PSL\n- Hujan, suhu, dan kelembapan bulanan, NASA POWER")
    st.subheader("Keterbatasan")
    st.markdown(
        "- Titik panas adalah indikator kebakaran, bukan kebakaran yang terkonfirmasi, dan dapat terhitung kurang "
        "saat tertutup awan atau asap.\n"
        "- Model bersifat prediktif dan tidak menyimpulkan sebab-akibat.\n"
        "- Fase El Niño dan La Niña memiliki sedikit episode pada data, sehingga skenario ENSO berbasis "
        "frekuensi historis yang sedikit pengamatannya.\n"
        "- Pembagian pulau memakai kotak koordinat perkiraan, bukan batas administratif.\n"
        "- Peta gambut dan tutupan lahan belum dimasukkan."
    )
    with st.expander("Daftar fitur model"):
        st.write(", ".join(M["fitur"]))


{"Peta Risiko": halaman_peta, "Skenario El Niño": halaman_skenario, "Riwayat vs Aktual": halaman_riwayat,
 "Evaluasi Model": halaman_evaluasi, "Tentang": halaman_tentang}[halaman]()
