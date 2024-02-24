import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso, ElasticNet, Ridge, LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
import seaborn as sns

# Fonksiyon: Performans Degerlerini Hesapla
def performance_calculate(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    return mae, mse, rmse, r2

class RegressionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Model Tahmin Arayüzü")

        # Veri Setini Okuyoruz
        self.veri = pd.read_csv("HavayoluYolcu.csv")  # "veri.csv" dosyanizin adini güncelleyin

        self.y = self.veri["YolcuSayisi"]
        self.veri.drop(["YolcuSayisi"], axis=1, inplace=True)
        self.x = self.veri

        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.x, self.y, train_size=0.3, random_state=46)

        self.scaler = StandardScaler()
        self.x_train_scaled = self.scaler.fit_transform(self.x_train)
        self.x_test_scaled = self.scaler.transform(self.x_test)

        # Algoritmalar
        self.algorithms = {
            "Lasso Regresyon": Lasso(alpha=0.1),
            "ElasticNet Regresyon": ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=10000, random_state=0),
            "Ridge Regresyon": Ridge(alpha=0.1),
            "Yapay Sinir Agi": MLPRegressor(hidden_layer_sizes=(200,90), learning_rate_init=0.04, max_iter=10000, random_state=46),
            "Linear Regression": LinearRegression()
        }

        # Dropdown Menü
        self.algorithm_var = tk.StringVar(root)
        self.algorithm_var.set("Yapay Sinir Agi")
        self.algorithm_menu = ttk.Combobox(root, textvariable=self.algorithm_var, values=list(self.algorithms.keys()))
        self.algorithm_menu.grid(row=0, column=0, padx=10, pady=10)

        # Tahmin Butonu
        self.predict_button = tk.Button(root, text="Tahmin Yap", command=self.predict_and_show_results)
        self.predict_button.grid(row=0, column=1, padx=10, pady=10)

        # Sonuçlar
        self.result_text_widget = tk.Text(root, height=15, width=80)
        self.result_text_widget.grid(row=1, column=0, columnspan=2, padx=10, pady=10)

    def predict_and_show_results(self):
        algorithm_name = self.algorithm_var.get()
        algorithm = self.algorithms[algorithm_name]
        algorithm.fit(self.x_train_scaled, self.y_train)
        predictions = algorithm.predict(self.x_test_scaled)

        performance = performance_calculate(self.y_test, predictions)

        result_text = f"\n{algorithm_name} Performans Degerleri:\n"
        metrics = ["Mean Absolute Error (MAE)", "Mean Squared Error (MSE)", "Root Mean Squared Error (RMSE)", "R2"]

        for metric, value in zip(metrics, performance):
            result_text += f"{metric}: {value}\n"

        result_text += "\nTahminler:\n"
        result_text += pd.DataFrame({"Gercek Deger": self.y_test, "Tahmin": predictions}).to_string()

        self.result_text_widget.delete(1.0, tk.END)  # Önceki içeriği temizle
        self.result_text_widget.insert(tk.END, result_text)

        # Regresyon grafiği çizimi için
        self.plot_regression_results(self.y_test, predictions, algorithm_name)

    def plot_regression_results(self, true_values, predicted_values, model_name):
        plt.scatter(true_values, predicted_values, color='blue')
        plt.plot([true_values.min(), true_values.max()], [true_values.min(), true_values.max()], 'k--', lw=2)
        plt.title(f"{model_name} Tahminleri")
        plt.xlabel("Gerçek Değerler")
        plt.ylabel("Tahmin Edilen Değerler")
        plt.show()

# Arayüzü Başlat
root = tk.Tk()
app = RegressionApp(root)
root.mainloop()
