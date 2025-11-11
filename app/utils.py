# app/utils.py
import pandas as pd
import random
import string
import time
from geopy.distance import geodesic
from geopy.geocoders import Nominatim

# Fungsi untuk menghitung jarak antar koordinat
def hitung_jarak(row):
    return geodesic(row["koordinat_dtw"], row["koordinat_user"]).kilometers

def preproses_koordinat(df):
    # --- fungsi untuk membaca koordinat berdasarkan nama lokasi ---
    def cek_koordinat(nama_string):
        # Generate 4 karakter acak (huruf dan angka)
        random_string = ''.join(random.choices(string.ascii_letters + string.digits, k=4))

        # Pilih delay acak antara 0.1 hingga 1.5 detik
        delay = random.uniform(0.1, 1.5)
        time.sleep(delay)

        # Inisialisasi geolocator
        geolocator = Nominatim(user_agent=random_string)

        try:
            lokasi = geolocator.geocode(nama_string)
            return lokasi.latitude, lokasi.longitude
        except:
            return 0, 0

    # --- ambil nilai unik asal (user) ---
    asal = df['asal (user)'].unique()
    df_asal = pd.DataFrame(asal, columns=['asal (user)'])
    df_asal['koordinat_user'] = df_asal['asal (user)'].apply(cek_koordinat)

    # Gabungkan koordinat DTW dari kolom latitude & longitude
    df["koordinat_dtw"] = df.apply(lambda row: f"({row.lattitude}, {row.longitude})", axis=1)

    # Petakan koordinat user ke df utama
    df = pd.merge(df, df_asal, on='asal (user)', how='left')

    # --- bersihkan koordinat invalid ---
    df['koordinat_user'] = df['koordinat_user'].apply(
        lambda x: (df['koordinat_user'].mean(), df['koordinat_user'].mean())
        if str(x) in ['(nan, nan)', '(°, °)', '((blank), (blank))'] else x
    )
    df['koordinat_dtw'] = df['koordinat_dtw'].apply(
        lambda x: (0, 0) if str(x) in ['(nan, nan)', '(°, °)', '((blank), (blank))'] else x
    )

    # --- konversi string koordinat ke tuple ---
    def convert_to_tuple(coord_str):
        if isinstance(coord_str, tuple):
            return coord_str
        if coord_str in ['(nan, nan)', '(°, °)', '((blank), (blank))']:
            return (0.0, 0.0)

        coord_str = coord_str.strip('()')
        lat, lon = coord_str.split(',')
        return (float(lat.replace('°', '').strip()), float(lon.replace('°', '').strip()))

    df['koordinat_dtw'] = df['koordinat_dtw'].apply(convert_to_tuple)
    df['koordinat_user'] = df['koordinat_user'].apply(convert_to_tuple)

    # --- hitung jarak antar titik ---
    df['jarak'] = df.apply(hitung_jarak, axis=1)
    return df

    import pandas as pd

def load_clean_csv(path):
    df = pd.read_csv(path, engine="python")

    # Normalisasi nama kolom
    df.columns = (
        df.columns
        .astype(str)
        .str.strip()
        .str.replace(" ", "_")
        .str.replace(",", "")
        .str.lower()
    )

    # Pastikan tipe data penting sesuai
    if "rating" in df.columns:
        df["rating"] = pd.to_numeric(df["rating"], errors="coerce").fillna(0)

    return df

