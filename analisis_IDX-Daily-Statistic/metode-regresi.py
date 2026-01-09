import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# Load dataset
print("="*50)
print("METODE 1: ANALISIS REGRESI LINIER & PERBANDINGAN STATISTIK")
print("="*50)

# Membaca dataset
df = pd.read_csv(r'C:\Code\Analisis\analisis_IDX-Daily-Statistic\dataset_idx_code_9.csv')  # Ganti dengan nama file yang sesuai

# Membersihkan dan mengubah format data
def clean_numeric(value):
    if isinstance(value, str):
        # Hapus titik yang digunakan sebagai pemisah ribuan
        value = value.replace('.', '')
        # Ganti koma dengan titik untuk desimal
        value = value.replace(',', '.')
        # Hapus simbol persen
        value = value.replace('%', '')
    try:
        return float(value)
    except:
        return np.nan

# Terapkan cleaning ke kolom numerik
df['Today_clean'] = df['Today'].apply(clean_numeric)
df['Change_clean'] = df['Change'].apply(clean_numeric)
df['Ytd Change_clean'] = df['Ytd Change'].apply(clean_numeric)

print("\nDataset Awal:")
print(df[['Country', 'Today', 'Change', 'Ytd Change']])

print("\nDataset Setelah Cleaning:")
print(df[['Country', 'Today_clean', 'Change_clean', 'Ytd Change_clean']])

# ANALISIS PERBANDINGAN 1: Perbandingan Statistik
print("\n" + "="*50)
print("ANALISIS PERBANDINGAN STATISTIK")
print("="*50)

# Statistik deskriptif
stats = df[['Today_clean', 'Change_clean', 'Ytd Change_clean']].describe()
print("\nStatistik Deskriptif:")
print(stats)

# Perbandingan antar negara
print("\nPerbandingan Nilai Today (Descending):")
today_comparison = df[['Country', 'Today_clean']].sort_values('Today_clean', ascending=False)
print(today_comparison)

print("\nPerbandingan Perubahan YTD (Descending):")
ytd_comparison = df[['Country', 'Ytd Change_clean']].sort_values('Ytd Change_clean', ascending=False)
print(ytd_comparison)

# ANALISIS PERBANDINGAN 2: Klasifikasi Performa
print("\n" + "="*50)
print("KLASIFIKASI PERFORMA")
print("="*50)

# Klasifikasi berdasarkan quartile
df['Today_Category'] = pd.qcut(df['Today_clean'], 3, labels=['Low', 'Medium', 'High'])
df['Ytd_Performance'] = np.where(df['Ytd Change_clean'] > 0, 'Positive', 'Negative')

print("\nKlasifikasi Berdasarkan Tingkat Today:")
print(df[['Country', 'Today_clean', 'Today_Category']])

print("\nKlasifikasi Berdasarkan Performa YTD:")
print(df[['Country', 'Ytd Change_clean', 'Ytd_Performance']])

# PREDIKSI: Regresi Linier untuk memprediksi YTD Change berdasarkan Today
print("\n" + "="*50)
print("PREDIKSI: REGRESI LINIER")
print("="*50)

# Persiapan data untuk prediksi
X = df[['Today_clean']].values
y = df['Ytd Change_clean'].values

# Model regresi linier
model = LinearRegression()
model.fit(X, y)

# Prediksi untuk semua negara
df['Predicted_Ytd'] = model.predict(X)

print("\nKoefisien Regresi:")
print(f"Slope: {model.coef_[0]:.4f}")
print(f"Intercept: {model.intercept_:.4f}")
print(f"R-squared: {model.score(X, y):.4f}")

print("\nHasil Prediksi vs Aktual:")
for idx, row in df.iterrows():
    print(f"{row['Country']}: Aktual={row['Ytd Change_clean']:.2f}%, Prediksi={row['Predicted_Ytd']:.2f}%")

# Prediksi untuk nilai Today baru
new_today_values = [8500, 8700, 9000]
print("\nPrediksi untuk nilai Today baru:")
for val in new_today_values:
    prediction = model.predict([[val]])[0]
    print(f"Today={val}: Prediksi YTD Change={prediction:.2f}%")

# Visualisasi
plt.figure(figsize=(15, 5))

# Subplot 1: Perbandingan Today
plt.subplot(1, 3, 1)
bars = plt.bar(df['Country'], df['Today_clean'])
plt.title('Perbandingan Nilai Today')
plt.xticks(rotation=45)
plt.ylabel('Today Value')
# Warna berdasarkan kategori
for bar, cat in zip(bars, df['Today_Category']):
    if cat == 'High':
        bar.set_color('green')
    elif cat == 'Medium':
        bar.set_color('orange')
    else:
        bar.set_color('red')

# Subplot 2: Perbandingan YTD Change
plt.subplot(1, 3, 2)
colors = ['green' if x > 0 else 'red' for x in df['Ytd Change_clean']]
plt.bar(df['Country'], df['Ytd Change_clean'], color=colors)
plt.title('Perbandingan YTD Change')
plt.xticks(rotation=45)
plt.ylabel('YTD Change (%)')
plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)

# Subplot 3: Regresi Linier
plt.subplot(1, 3, 3)
plt.scatter(df['Today_clean'], df['Ytd Change_clean'], color='blue', label='Aktual')
plt.plot(df['Today_clean'], df['Predicted_Ytd'], color='red', label='Prediksi')
plt.title('Regresi Linier: Today vs YTD Change')
plt.xlabel('Today Value')
plt.ylabel('YTD Change (%)')
plt.legend()

plt.tight_layout()
plt.show()

# Ringkasan
print("\n" + "="*50)
print("RINGKASAN ANALISIS")
print("="*50)
print(f"1. Negara dengan Today tertinggi: {df.loc[df['Today_clean'].idxmax(), 'Country']}")
print(f"2. Negara dengan YTD Change tertinggi: {df.loc[df['Ytd Change_clean'].idxmax(), 'Country']}")
print(f"3. Rata-rata Today: {df['Today_clean'].mean():.2f}")
print(f"4. Rata-rata YTD Change: {df['Ytd Change_clean'].mean():.2f}%")
print(f"5. Korelasi Today-YTD: {df['Today_clean'].corr(df['Ytd Change_clean']):.4f}")