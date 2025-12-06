# Import library
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
import joblib
import os

# Path otomatis agar bekerja di local & GitHub Actions
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
raw_file = os.path.join(BASE_DIR, "obesity_classification_raw.csv")
output_folder = os.path.join(BASE_DIR, "preprocessing")
output_file = os.path.join(output_folder, "obesity_classification_preprocessing.csv")

# Load Dataset
df = pd.read_csv(raw_file)
print("Dataset berhasil dibaca, contoh data:")
print(df.head())

# Menentukan kolom numerik dan kategorikal
numerical_cols = df.select_dtypes(include=['number']).columns
categorical_cols = df.select_dtypes(include=['object']).columns

# Menghapus Duplikat
df_cleaned = df.drop_duplicates()

# Encoding LabelEncoder Kolom Kategorikal
label_encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    df_cleaned[col] = le.fit_transform(df_cleaned[col])
    label_encoders[col] = le

# Simpan Label Encoders
label_encoder_path = os.path.join(output_folder, "label_encoders.pkl")
joblib.dump(label_encoders, label_encoder_path)
print(f"Label Encoders disimpan di: {label_encoder_path}")

# Scaling kolom Numerik dengan ColumnTransformer
scaler = StandardScaler()
preprocessor = ColumnTransformer(
    transformers=[
        ('num', scaler, numerical_cols)
    ],
    remainder='drop'
)

# Menjalankan preprocessing scaling
data_scaled = preprocessor.fit_transform(df_cleaned[numerical_cols])

# Simpan Scaler
preprocessor_path = os.path.join(output_folder, "preprocessor.pkl")
joblib.dump(preprocessor, preprocessor_path)
print(f"Preprocessor (scaler) disimpan di: {preprocessor_path}")

# Menggabungkan kolom numerik dan kategorikal
df_preprocessed = df_cleaned.copy()
df_preprocessed[numerical_cols] = data_scaled

# Simpan hasil dalam format csv
df_preprocessed.to_csv(output_file, index=False)
print(f"Hasil preprocessing disimpan di: {output_file}")