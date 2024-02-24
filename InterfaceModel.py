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
from tkinter import filedialog, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Veri Setini Okuma Fonksiyonu
def read_data():
    file_path = filedialog.askopenfilename(title="Veri Seti Seçin", filetypes=[("CSV files", "*.csv")])
    if file_path:
        return pd.read_csv(file_path)
    else:
        return None

# Fonksiyon: Regresyon Modeli Oluştur ve Tahmin Yap
def build_and_predict_model(model, x_train, x_test, y_train):
    model.fit(x_train, y_train)
    tahmin = model.predict(x_test)
    return tahmin

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
        self.root.title("Regresyon Analizi Uygulaması")

        # Veri Setini Oku Butonu
        self.load_data_button = tk.Button(root, text="Veri Setini Yükle", command=self.load_data)
        self.load_data_button.pack(pady=10)

        # Analiz Yap Butonu
        self.analyze_button = tk.Button(root, text="Analiz Yap", command=self.perform_analysis)
        self.analyze_button.pack(pady=10)

        # Sonuçları Göster Butonu
        self.show_results_button = tk.Button(root, text="Sonuçları Göster", command=self.show_results)
        self.show_results_button.pack(pady=10)

        # Grafik Alanı
        self.fig, self.ax = plt.subplots()
        self.canvas = FigureCanvasTkAgg(self.fig, master=root)
        self.canvas.get_tk_widget().pack()

        self.veri = None
        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.y_test = None
        self.predictions = None
        self.regressors = [
            ("Lasso Regresyon", Lasso(alpha=0.1)),
            ("ElasticNet Regresyon", ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=10000, random_state=0)),
            ("Ridge Regresyon", Ridge(alpha=0.1)),
            ("Yapay Sinir Agi", MLPRegressor(hidden_layer_sizes=(200, 100), learning_rate_init=0.04, max_iter=10000, random_state=46)),
            ("Linear Regression", LinearRegression())
        ]

    def load_data(self):
        self.veri = read_data()
        if self.veri is not None:
            messagebox.showinfo("Bilgi", "Veri seti başarıyla yüklendi!")

    def perform_analysis(self):
        if self.veri is not None:
            # Veri Setini Ayırma
            y = self.veri["YolcuSayisi"]
            self.veri.drop(["YolcuSayisi"], axis=1, inplace=True)
            x = self.veri

            self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(x, y, train_size=0.3, random_state=46)

            self.predictions = {}
            for name, regressor in self.regressors:
                tahmin = build_and_predict_model(regressor, self.x_train, self.x_test, self.y_train)
                self.predictions[name] = tahmin

            messagebox.showinfo("Bilgi", "Analiz başarıyla tamamlandı!")

    def show_results(self):
        if self.predictions is not None:
            self.ax.clear()

            # Her bir model için regresyon grafikleri çizimi
            for name, prediction in self.predictions.items():
                self.ax.scatter(self.y_test, prediction, label=name)

            self.ax.plot([self.y_test.min(), self.y_test.max()], [self.y_test.min(), self.y_test.max()], 'k--', lw=2)
            self.ax.set_title("Regresyon Modellerinin Tahminleri")
            self.ax.set_xlabel("Gerçek Değerler")
            self.ax.set_ylabel("Tahmin Edilen Değerler")
            self.ax.legend()

            self.canvas.draw()
        else:
            messagebox.showwarning("Uyarı", "Lütfen önce bir analiz yapın!")

if __name__ == "__main__":
    root = tk.Tk()
    app = RegressionApp(root)
    root.mainloop()

"""asdsadsa"""