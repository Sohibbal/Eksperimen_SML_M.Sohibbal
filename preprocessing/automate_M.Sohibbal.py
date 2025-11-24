import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
import os

# Path otomatis agar bekerja di local & GitHub Actions
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
raw_file = os.path.join(BASE_DIR, "obesity_classification_raw.csv")
output_folder = os.path.join(BASE_DIR, "preprocessing")
output_file = os.path.join(output_folder, "obesity_classification_preprocessing.csv")

# Dataset
df = pd.read_csv(raw_file)
print("Dataset berhasil dibaca, contoh data:")
print(df.head())

# Menentukan kolom numerik dan kategorikal
numerical_cols = df.select_dtypes(include=['number']).columns
categorical_cols = df.select_dtypes(include=['object']).columns

# Menghapus Duplikat
df_cleaned = df.drop_duplicates()

# ======================================================================
# 1️⃣ Encoding LabelEncoder UNTUK SETIAP KOLOM KATEGORIKAL
# ======================================================================
label_encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    df_cleaned[col] = le.fit_transform(df_cleaned[col])
    label_encoders[col] = le   # simpan encoder jika nanti dibutuhkan
# ======================================================================

# ======================================================================
# 2️⃣ Scaling numerik saja dengan ColumnTransformer
# ======================================================================
scaler = StandardScaler()

preprocessor = ColumnTransformer(
    transformers=[
        ('num', scaler, numerical_cols)
    ],
    remainder='drop'   # kolom kategorikal sudah di-encode → tidak perlu transformer
)

# Menjalankan preprocessing
data_scaled = preprocessor.fit_transform(df_cleaned[numerical_cols])

# ======================================================================
# 3️⃣ Bentuk DataFrame final → gabungkan kolom kategori + hasil scaling
# ======================================================================
df_preprocessed = df_cleaned.copy()
df_preprocessed[numerical_cols] = data_scaled

# Simpan hasil
df_preprocessed.to_csv(output_file, index=False)
print(f"Hasil preprocessing disimpan di: {output_file}")

# trigger workflow