import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import seaborn as sns
import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Fonksiyon: Performans Degerlerini Hesapla
def performance_calculate(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    return mae, mse, rmse, r2

# Veri Setini Okuyoruz
veri = pd.read_csv("HavayoluYolcu.csv")  # "veri.csv" dosyanizin adini güncelleyin

y = veri["YolcuSayisi"]
veri.drop(["YolcuSayisi"], axis=1, inplace=True)
x = veri

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.3, random_state=46)

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

# Tkinter Arayüzü için
root = tk.Tk()
root.title("Regresyon Tahmin Arayüzü")


# Algoritmalar
algorithms = {
    "Linear Regression": LinearRegression(),
    "Lasso Regression": Lasso(alpha=0.1),
    "Ridge Regression": Ridge(alpha=0.1),
    "ElasticNet Regression": ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=10000, random_state=0),
    "Yapay Sinir Agi": MLPRegressor(hidden_layer_sizes=(200, 100), learning_rate_init=0.04, max_iter=10000, random_state=46)
}

# Performans Degerlerini Hesapla
def calculate_performance(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    return mae, mse, rmse, r2

# Tahmin ve Performansı Göster
def predict_and_show_results():
    algorithm_name = algorithm_var.get()
    algorithm = algorithms[algorithm_name]
    algorithm.fit(x_train_scaled, y_train)
    predictions = algorithm.predict(x_test_scaled)

    performance = calculate_performance(y_test, predictions)

    result_text = f"\n{algorithm_name} Performans Degerleri:\n"
    metrics = ["Mean Absolute Error (MAE)", "Mean Squared Error (MSE)", "Root Mean Squared Error (RMSE)", "R2"]

    for metric, value in zip(metrics, performance):
        result_text += f"{metric}: {value}\n"

    result_text += "\nTahminler:\n"
    result_text += pd.DataFrame({"Gerçek Deger": y_test, "Tahmin": predictions}).to_string()

    result_text_widget.delete(1.0, tk.END)  # Önceki içeriği temizle
    result_text_widget.insert(tk.END, result_text)

    # Regresyon grafiği çizimi
    plt.figure()
    plt.scatter(y_test, predictions, color='blue')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
    plt.title(f"{algorithm_name} Tahminleri")
    plt.xlabel("Gerçek Değerler")
    plt.ylabel("Tahmin Edilen Değerler")

    # Grafiği Tkinter arayüzüne entegre et
    canvas = FigureCanvasTkAgg(plt.gcf(), master=root)
    canvas_widget = canvas.get_tk_widget()
    canvas_widget.grid(row=1, column=2, padx=10, pady=10)

# Dropdown Menü
algorithm_var = tk.StringVar(root)
algorithm_var.set("Linear Regression")
algorithm_menu = ttk.Combobox(root, textvariable=algorithm_var, values=list(algorithms.keys()))
algorithm_menu.grid(row=0, column=0, padx=10, pady=10)

# Tahmin Butonu
predict_button = tk.Button(root, text="Tahmin Yap", command=predict_and_show_results)
predict_button.grid(row=0, column=1, padx=10, pady=10)

# Sonuçlar
result_text_widget = tk.Text(root, height=15, width=50)
result_text_widget.grid(row=1, column=0, columnspan=2, padx=10, pady=10)

# Arayüzü Başlat
root.mainloop()
