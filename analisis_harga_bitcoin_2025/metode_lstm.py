import os
print(os.getcwd())
import pandas as pd
import numpy as np
import math
import torch
from torch import nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

# --------------------------
# 1. Fungsi Windowing
# --------------------------
def create_time_series_window(data, look_back=1):
    X, Y = [], []
    for i in range(len(data) - look_back - 1):
        X.append(data[i:(i + look_back), 0])
        Y.append(data[i + look_back, 0])
    return np.array(X), np.array(Y)

# --------------------------
# 2. Model LSTM PyTorch
# --------------------------
class LSTMModel(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=50, num_layers=2, output_dim=1):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim)
        
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])  # ambil output timestep terakhir
        return out

# --------------------------
# 3. Evaluasi
# --------------------------
def evaluate_model(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return mae, rmse, mape

# --------------------------
# 4. MAIN (RUN ANALYSIS)
# --------------------------
def run_lstm_pytorch(file_path):
    # a. Load Data
    df = pd.read_csv(file_path)
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
    df.set_index('Date', inplace=True)

    df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
    df = df.dropna(subset=['Close'])


    data = df['Close'].values.reshape(-1, 1)

    # b. Normalisasi
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    # c. Train-Test Split (80:20)
    training_data_len = math.ceil(len(scaled_data) * 0.8)
    train_data = scaled_data[:training_data_len]
    test_data = scaled_data[training_data_len - 60:]

    # d. Windowing
    look_back = 60
    X_train, y_train = create_time_series_window(train_data, look_back)
    X_test, y_test = create_time_series_window(test_data, look_back)

    # e. Reshape untuk PyTorch (samples, timesteps, features)
    X_train = torch.tensor(X_train, dtype=torch.float32).reshape(-1, look_back, 1)
    y_train = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)
    X_test = torch.tensor(X_test, dtype=torch.float32).reshape(-1, look_back, 1)

    # f. Model
    model = LSTMModel()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # g. Training
    epochs = 5
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        output = model(X_train)
        loss = criterion(output, y_train)

        if torch.isnan(loss):
            print("⚠️ TERDETEKSI NaN — hentikan training!")
            break

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.6f}")


    # h. Prediksi
    model.eval()
    predictions_scaled = model(X_test).detach().cpu().numpy()


    # i. Inverse transform to original scale
    dummy_array = np.zeros((len(predictions_scaled), 1))
    dummy_array[:, 0] = predictions_scaled[:, 0]
    predictions = scaler.inverse_transform(dummy_array)[:, 0]

    y_test_original = data[training_data_len + look_back + 1:].reshape(-1)

    # j. Evaluasi
    mae, rmse, mape = evaluate_model(y_test_original, predictions)

    results = {
        'Model': 'PyTorch-LSTM',
        'MAE': mae,
        'RMSE': rmse,
        'MAPE': mape
    }

    return results, y_test_original, predictions


# --------------------------
# 5. RUN
# --------------------------
file_path = r'C:\Code\Analisis\analisis_harga_bitcoin_2025\dataset_btc.csv'
results, actual, pred = run_lstm_pytorch(file_path)

print("\n===== HASIL ANALISIS (PyTorch) =====")
print(results)
