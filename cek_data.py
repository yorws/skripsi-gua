import pandas as pd
import os

# Ini buat nyari alamat folder tempat file ini berada
base_dir = os.path.dirname(os.path.abspath(__file__))
print(f"Lokasi folder lu sekarang: {base_dir}")

# Kita coba gabungin alamatnya
path_csv = os.path.join(base_dir, 'dataset', 'ml-latest-small', 'ratings.csv')

print(f"Lagi nyoba buka: {path_csv}")

try:
    df = pd.read_csv(path_csv)
    print("MANTAP! Datanya keluar:")
    print(df.head())
except Exception as e:
    print(f"Masih belum ketemu kawan. Error: {e}")
    print("Coba cek di Explorer sebelah kiri, folder 'dataset' ada di mana?")