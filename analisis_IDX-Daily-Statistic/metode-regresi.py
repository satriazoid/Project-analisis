import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

print("="*50)
print("METODE 1: ANALISIS REGRESI LINIER & PERBANDINGAN STATISTIK")
print("="*50)

df = pd.read_csv(r'C:\Code\Analisis\analisis_IDX-Daily-Statistic\dataset_idx_code_9.csv')  # Ganti dengan nama file yang sesuai

def clean_numeric(value):
    if isinstance(value, str):
        value = value.replace('.', '')
        value = value.replace(',', '.')
        value = value.replace('%', '')
    try:
        return float(value)
    except:
        return np.nan

df['Today_clean'] = df['Today'].apply(clean_numeric)
df['Change_clean'] = df['Change'].apply(clean_numeric)
df['Ytd Change_clean'] = df['Ytd Change'].apply(clean_numeric)

print("\nDataset Awal:")
print(df[['Country', 'Today', 'Change', 'Ytd Change']])
print(df[['Country', 'Today_clean', 'Change_clean', 'Ytd Change_clean']])

print("\n" + "="*50)
print("ANALISIS PERBANDINGAN STATISTIK")
print("="*50)

stats = df[['Today_clean', 'Change_clean', 'Ytd Change_clean']].describe()
print("\nStatistik Deskriptif:")
print(stats)

print("\nPerbandingan Nilai Today (Descending):")
today_comparison = df[['Country', 'Today_clean']].sort_values('Today_clean', ascending=False)
print(today_comparison)

print("\nPerbandingan Perubahan YTD (Descending):")
ytd_comparison = df[['Country', 'Ytd Change_clean']].sort_values('Ytd Change_clean', ascending=False)
print(ytd_comparison)

print("\n" + "="*50)
print("KLASIFIKASI PERFORMA")
print("="*50)

df['Today_Category'] = pd.qcut(df['Today_clean'], 3, labels=['Low', 'Medium', 'High'])
df['Ytd_Performance'] = np.where(df['Ytd Change_clean'] > 0, 'Positive', 'Negative')

print("\nKlasifikasi Berdasarkan Tingkat Today:")
print(df[['Country', 'Today_clean', 'Today_Category']])

print("\nKlasifikasi Berdasarkan Performa YTD:")
print(df[['Country', 'Ytd Change_clean', 'Ytd_Performance']])

print("\n" + "="*50)
print("PREDIKSI: REGRESI LINIER")
print("="*50)

X = df[['Today_clean']].values
y = df['Ytd Change_clean'].values

model = LinearRegression()
model.fit(X, y)

df['Predicted_Ytd'] = model.predict(X)

print("\nKoefisien Regresi:")
print(f"Slope: {model.coef_[0]:.4f}")
print(f"Intercept: {model.intercept_:.4f}")
print(f"R-squared: {model.score(X, y):.4f}")

print("\nHasil Prediksi vs Aktual:")
for idx, row in df.iterrows():
    print(f"{row['Country']}: Aktual={row['Ytd Change_clean']:.2f}%, Prediksi={row['Predicted_Ytd']:.2f}%")

new_today_values = [8500, 8700, 9000]
print("\nPrediksi untuk nilai Today baru:")
for val in new_today_values:
    prediction = model.predict([[val]])[0]
    print(f"Today={val}: Prediksi YTD Change={prediction:.2f}%")

plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
bars = plt.bar(df['Country'], df['Today_clean'])
plt.title('Perbandingan Nilai Today')
plt.xticks(rotation=45)
plt.ylabel('Today Value')
for bar, cat in zip(bars, df['Today_Category']):
    if cat == 'High':
        bar.set_color('green')
    elif cat == 'Medium':
        bar.set_color('orange')
    else:
        bar.set_color('red')

plt.subplot(1, 3, 2)
colors = ['green' if x > 0 else 'red' for x in df['Ytd Change_clean']]
plt.bar(df['Country'], df['Ytd Change_clean'], color=colors)
plt.title('Perbandingan YTD Change')
plt.xticks(rotation=45)
plt.ylabel('YTD Change (%)')
plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)

plt.subplot(1, 3, 3)
plt.scatter(df['Today_clean'], df['Ytd Change_clean'], color='blue', label='Aktual')
plt.plot(df['Today_clean'], df['Predicted_Ytd'], color='red', label='Prediksi')
plt.title('Regresi Linier: Today vs YTD Change')
plt.xlabel('Today Value')
plt.ylabel('YTD Change (%)')
plt.legend()

plt.tight_layout()
plt.show()

print("\n" + "="*50)
print("RINGKASAN ANALISIS")
print("="*50)
print(f"1. Negara dengan Today tertinggi: {df.loc[df['Today_clean'].idxmax(), 'Country']}")
print(f"2. Negara dengan YTD Change tertinggi: {df.loc[df['Ytd Change_clean'].idxmax(), 'Country']}")
print(f"3. Rata-rata Today: {df['Today_clean'].mean():.2f}")
print(f"4. Rata-rata YTD Change: {df['Ytd Change_clean'].mean():.2f}%")
print(f"5. Korelasi Today-YTD: {df['Today_clean'].corr(df['Ytd Change_clean']):.4f}")