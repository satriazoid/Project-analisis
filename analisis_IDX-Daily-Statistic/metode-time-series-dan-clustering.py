import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')

# Load dataset
print("="*50)
print("METODE 2: ANALISIS CLUSTERING & TIME SERIES")
print("="*50)

# Membaca dataset
df = pd.read_csv(r'C:\Code\Analisis\analisis_IDX-Daily-Statistic\dataset_idx_code_9.csv')  # Ganti dengan nama file yang sesuai

# Fungsi untuk cleaning data
def clean_numeric(value):
    if isinstance(value, str):
        value = value.replace('.', '')
        value = value.replace(',', '.')
        value = value.replace('%', '')
    try:
        return float(value)
    except:
        return np.nan

# Terapkan cleaning
df['Today_clean'] = df['Today'].apply(clean_numeric)
df['Change_clean'] = df['Change'].apply(clean_numeric)
df['Ytd Change_clean'] = df['Ytd Change'].apply(clean_numeric)

print("\nDataset yang digunakan:")
print(df[['Country', 'Today_clean', 'Change_clean', 'Ytd Change_clean']])

# ANALISIS PERBANDINGAN 1: Clustering K-Means
print("\n" + "="*50)
print("ANALISIS CLUSTERING (K-MEANS)")
print("="*50)

# Persiapan data untuk clustering
X = df[['Today_clean', 'Ytd Change_clean']].values

# Standardisasi data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Menentukan jumlah cluster optimal menggunakan silhouette score
silhouette_scores = []
k_range = range(2, 6)

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    score = silhouette_score(X_scaled, kmeans.labels_)
    silhouette_scores.append(score)

# Pilih k dengan silhouette score tertinggi
optimal_k = k_range[np.argmax(silhouette_scores)]

# Clustering dengan k optimal
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
df['Cluster'] = kmeans.fit_predict(X_scaled)

print(f"\nJumlah cluster optimal: {optimal_k}")
print("\nHasil Clustering:")
for cluster in sorted(df['Cluster'].unique()):
    cluster_countries = df[df['Cluster'] == cluster]['Country'].tolist()
    print(f"\nCluster {cluster}:")
    print(f"  Negara: {', '.join(cluster_countries)}")
    cluster_data = df[df['Cluster'] == cluster]
    print(f"  Rata-rata Today: {cluster_data['Today_clean'].mean():.2f}")
    print(f"  Rata-rata YTD Change: {cluster_data['Ytd Change_clean'].mean():.2f}%")

# ANALISIS PERBANDINGAN 2: Analisis Tren dan Momentum
print("\n" + "="*50)
print("ANALISIS TREN DAN MOMENTUM")
print("="*50)

# Hitung momentum (kombinasi Change dan YTD Change)
df['Momentum_Score'] = (df['Change_clean'] * 0.3) + (df['Ytd Change_clean'] * 0.7)
df['Trend'] = np.where(df['Momentum_Score'] > 0, 'Uptrend', 'Downtrend')

print("\nAnalisis Momentum:")
momentum_df = df[['Country', 'Change_clean', 'Ytd Change_clean', 'Momentum_Score', 'Trend']].sort_values('Momentum_Score', ascending=False)
print(momentum_df)

# Klasifikasi berdasarkan momentum
df['Momentum_Category'] = pd.qcut(df['Momentum_Score'], 4, 
                                   labels=['Very Weak', 'Weak', 'Strong', 'Very Strong'])

print("\nKlasifikasi Momentum:")
print(df[['Country', 'Momentum_Score', 'Momentum_Category', 'Trend']])

# PREDIKSI: Metode Weighted Average untuk prediksi YTD Change
print("\n" + "="*50)
print("PREDIKSI: METODE WEIGHTED AVERAGE")
print("="*50)

# Prediksi menggunakan weighted average dari cluster
def predict_ytd(country, today_value):
    # Cari cluster negara tersebut
    country_cluster = df[df['Country'] == country]['Cluster'].values[0]
    
    # Hitung rata-rata YTD Change di cluster tersebut
    cluster_avg_ytd = df[df['Cluster'] == country_cluster]['Ytd Change_clean'].mean()
    
    # Hitung rata-rata Today di cluster tersebut
    cluster_avg_today = df[df['Cluster'] == country_cluster]['Today_clean'].mean()
    
    # Prediksi dengan adjustment berdasarkan deviasi dari rata-rata cluster
    today_deviation = (today_value - cluster_avg_today) / cluster_avg_today
    predicted_ytd = cluster_avg_ytd * (1 + today_deviation * 0.5)  # Faktor adjustment
    
    return predicted_ytd

# Prediksi untuk setiap negara dengan asumsi Today naik 1%
print("\nPrediksi YTD Change jika Today naik 1%:")
predictions = []
for idx, row in df.iterrows():
    new_today = row['Today_clean'] * 1.01  # Today naik 1%
    predicted_ytd = predict_ytd(row['Country'], new_today)
    predictions.append({
        'Country': row['Country'],
        'Current_Today': row['Today_clean'],
        'New_Today': new_today,
        'Current_YTD': row['Ytd Change_clean'],
        'Predicted_YTD': predicted_ytd,
        'Change_in_YTD': predicted_ytd - row['Ytd Change_clean']
    })

prediction_df = pd.DataFrame(predictions)
print(prediction_df)

# Prediksi untuk negara baru (contoh)
print("\nPrediksi untuk negara baru:")
new_countries = [
    {'Country': 'New_Country_A', 'Today': 8600},
    {'Country': 'New_Country_B', 'Today': 8300},
    {'Country': 'New_Country_C', 'Today': 8900}
]

for new_country in new_countries:
    # Temukan cluster terdekat berdasarkan nilai Today
    today_diff = np.abs(df['Today_clean'] - new_country['Today'])
    nearest_cluster = df.loc[today_diff.idxmin(), 'Cluster']
    
    # Prediksi berdasarkan cluster terdekat
    cluster_avg = df[df['Cluster'] == nearest_cluster]['Ytd Change_clean'].mean()
    print(f"{new_country['Country']} (Today={new_country['Today']}): Prediksi YTD ≈ {cluster_avg:.2f}% (Cluster {nearest_cluster})")

# Visualisasi
plt.figure(figsize=(15, 5))

# Subplot 1: Hasil Clustering
plt.subplot(1, 3, 1)
colors = ['red', 'blue', 'green', 'purple', 'orange']
for cluster in sorted(df['Cluster'].unique()):
    cluster_data = df[df['Cluster'] == cluster]
    plt.scatter(cluster_data['Today_clean'], cluster_data['Ytd Change_clean'], 
                color=colors[cluster], label=f'Cluster {cluster}', s=100)
    
    # Tambah label negara
    for idx, row in cluster_data.iterrows():
        plt.annotate(row['Country'], 
                    (row['Today_clean'], row['Ytd Change_clean']),
                    textcoords="offset points",
                    xytext=(0,10),
                    ha='center',
                    fontsize=8)

plt.title('Clustering Negara Berdasarkan Today dan YTD Change')
plt.xlabel('Today Value')
plt.ylabel('YTD Change (%)')
plt.legend()
plt.grid(True, alpha=0.3)

# Subplot 2: Analisis Momentum
plt.subplot(1, 3, 2)
colors_momentum = ['red' if x == 'Downtrend' else 'green' for x in df['Trend']]
bars = plt.bar(df['Country'], df['Momentum_Score'], color=colors_momentum)
plt.title('Skor Momentum per Negara')
plt.xticks(rotation=45)
plt.ylabel('Momentum Score')
plt.axhline(y=0, color='black', linestyle='-', alpha=0.5)

# Subplot 3: Perbandingan Hasil Prediksi
plt.subplot(1, 3, 3)
x = range(len(prediction_df))
width = 0.35
plt.bar([i - width/2 for i in x], prediction_df['Current_YTD'], width, label='Aktual', alpha=0.7)
plt.bar([i + width/2 for i in x], prediction_df['Predicted_YTD'], width, label='Prediksi', alpha=0.7)
plt.title('Perbandingan YTD Aktual vs Prediksi')
plt.xticks(x, prediction_df['Country'], rotation=45)
plt.ylabel('YTD Change (%)')
plt.legend()

plt.tight_layout()
plt.show()

# Ringkasan
print("\n" + "="*50)
print("RINGKASAN ANALISIS METODE 2")
print("="*50)
print(f"1. Jumlah cluster optimal: {optimal_k}")
print(f"2. Negara dengan momentum terkuat: {df.loc[df['Momentum_Score'].idxmax(), 'Country']}")
print(f"3. Persentase negara dengan trend positif: {(df['Trend'] == 'Uptrend').sum() / len(df) * 100:.1f}%")
print(f"4. Rata-rata momentum: {df['Momentum_Score'].mean():.2f}")
print(f"5. Standar deviasi momentum: {df['Momentum_Score'].std():.2f}")