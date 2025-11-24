import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import os

# Path otomatis agar bekerja di local & GitHub Actions
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
raw_file = os.path.join(BASE_DIR, "obesity_classification_raw.csv")
output_folder = os.path.join(BASE_DIR, "preprocessing")
output_file = os.path.join(output_folder, "obesity_classification_preprocessed.csv")

# Dataset
df = pd.read_csv(raw_file)
print("Dataset berhasil dibaca, contoh data:")
print(df.head())

# Menentukan kolom numerik dan kategorikal
numerical_cols = df.select_dtypes(include=['number']).columns
categorical_cols = df.select_dypes(include=['object']).columns

# Menghapus Duplikat
df_cleaned = df.drop_duplicates()

# Preprocessing
scaler = StandardScaler()
encoder = OneHotEncoder(sparse_output=False, drop='first')

# Column Transformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', scaler, numerical_cols),
        ('cat', encoder, categorical_cols)
    ]
)

# Menjalankan preprocessing
data_preprocessed = preprocessor.fit_transform(df_cleaned)

# Mengambil nama kolom hasil encoding
encoded_cols = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_cols)

# Membuat list semua kolom final
final_columns = list(numerical_cols) + list(encoded_cols)

# Membuat DataFrame final
df_preprocessed = pd.DataFrame(data_preprocessed, columns=final_columns)

# Simpan hasil
df_preprocessed.to_csv(output_file, index=False)
print(f"Hasil preprocessing disimpan di: {output_file}")

# trigger workflow
