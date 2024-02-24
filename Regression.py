import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import seaborn as sns

# Fonksiyon: Performans Degerlerini Hesapla
def performance_calculate(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    return mae, mse, rmse, r2

#
# Veri Setini Okuma İşlemi
veri = pd.read_csv("HavayoluYolcu.csv")  

#Bağımlı Değişken Seçimi
y = veri["YolcuSayisi"]
veri.drop(["YolcuSayisi"], axis=1, inplace=True)
x = veri

#Veri Setinin Bölünmesi
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.3, random_state=46
)

#Normalizasyon
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)   #Eğitim veri setindeki özellikleri standartlaştırma işlemi 
x_test_scaled = scaler.transform(x_test)         #Test --> Önceden hesaplanan standartlaştırma değerleri kullanarak dönüşüm yapıldı. 

# LINEAR REGRESSION ALGORİTMASI
linear_reg = LinearRegression()
linear_reg.fit(x_train, y_train)   #Eğitim veri seti için model eğitimi 
tahmin_linear_reg = linear_reg.predict(x_test)   #Eğitilmiş modeli kullanarak tahmin yapma 

# RIDGE REGRESYON ALGORİTMASI
ridge_reg = Ridge(alpha=0.1) 
ridge_reg.fit(x_train, y_train)
tahmin_ridge = ridge_reg.predict(x_test)
#

# LASSO REGRESYON ALGORİTMASI
lassoReg = Lasso(alpha=0.1)
lassoReg.fit(x_train, y_train)
tahmin_lasso = lassoReg.predict(x_test)

# ELASTİC REGRESYON ALGORİTMASI
elastic_reg = ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=10000, random_state=0)
elastic_reg.fit(x_train, y_train)
tahmin_elastic = elastic_reg.predict(x_test)

# YAPAY SİNİR AgI ALGORİTMASI
mlp_reg = MLPRegressor(
    hidden_layer_sizes=(200, 100),
    learning_rate_init=0.04,
    max_iter=10000,
    random_state=46,
)
#Eğitim ve tahmin işlemi
mlp_reg.fit(x_train, y_train)
tahmin_mlp_reg = mlp_reg.predict(x_test)

#
# Performans Degerlendirmeleri ve Tablosu
predicts = [                
    tahmin_lasso,
    tahmin_elastic,
    tahmin_ridge,
    tahmin_mlp_reg,
    tahmin_linear_reg,
]
algoritma_names = [        
    "Lasso Regresyon",
    "ElasticNet Regresyon",
    "Ridge Regresyon",
    "Yapay Sinir Agi",
    "Linear Regression",
]

# LINEAR REGRESSION TAHMİNLERİ


linear_tahmin = pd.DataFrame(
    {"Gercek Deger": y_test, "Linear Regression Tahmini": tahmin_linear_reg}
)
# RIDGE REGRESYON TAHMİNLERİ
ridge_tahmin = pd.DataFrame({"Gercek Deger": y_test, "Ridge Tahmini": tahmin_ridge})
#

# Yapay Sinir Agi Tahminleri
mlp_tahmin = pd.DataFrame(
    {"Gercek Deger": y_test, "Yapay Sinir Agi Tahmini": tahmin_mlp_reg}
)
print("\nYapay Sinir Agi Tahminleri ve Gercek Degerler:")
print(mlp_tahmin.to_string())

# LASSO REGRESYON TAHMİNLERİ
lasso_tahmin = pd.DataFrame({"Gercek Deger": y_test, "Lasso Tahmini": tahmin_lasso})

# ELASTİC REGRESYON TAHMİNLERİ
elastic_tahmin = pd.DataFrame(
    {"Gercek Deger": y_test, "ElasticNet Tahmini": tahmin_elastic}
)
# Tahmin ve Gercek Degerleri Gösterilmesi

print("\nLasso Tahminleri ve Gercek Degerler:")
print(lasso_tahmin.to_string())

print("\nElasticNet Tahminleri ve Gercek Degerler:")
print(elastic_tahmin.to_string())

print("\nRidge Tahminleri ve Gercek Degerler:")
print(ridge_tahmin.to_string())

print("\nLinear Regression Tahminleri ve Gercek Degerler:")
print(linear_tahmin.to_string())

print("\nYapay Sinir Agi Tahminleri ve Gercek Degerler:")
print(mlp_tahmin.to_string())

#
#Seriler
seriler = []
metrics = [
    "Mean Absolute Error (MAE)",
    "Mean Squared Error (MSE)",
    "Root Mean Squared Error (RMSE)",
    "R2",
]
#Performans Ölçütleri
for i, predict in enumerate(predicts):
    data = performance_calculate(y_test, predict)  #Performans metriklerini hesaplar
    seriler.append(data)  

#Seriler listesindeki performans verileri ile pandas dataframe oluşturma
df = pd.DataFrame(data=seriler, index=algoritma_names, columns=metrics)
pd.set_option("display.colheader_justify", "center")  


print("\n\nPerformans Tablosu:\n")
print(df.to_string())  #DataFrame yazdırma

# Regresyon grafiği çizimi için
def plot_regression_results(true_values, predicted_values, model_name):
    plt.scatter(true_values, predicted_values, color="blue")   #Gerçek ve tahmini değerler arasındaki ilişki 
    plt.plot(
        [true_values.min(), true_values.max()],
        [true_values.min(), true_values.max()],
        "k--",  #siyah kesikli çizgi
        lw=2,   #45 derece çizgisi
    )
    plt.title(f"{model_name} Tahminleri")
    plt.xlabel("Gerçek Değerler")
    plt.ylabel("Tahmin Edilen Değerler")
    plt.show()    #Grafik gösterimi 
#

# Model isimleri
model_names = [
    "Lasso Regresyon",
    "ElasticNet Regresyon",
    "Ridge Regresyon",
    "Yapay Sinir Agi",
    "Linear Regression",
]

# Her bir model için regresyon grafikleri çizimi
for i, predict in enumerate(predicts):
    plot_regression_results(y_test, predict, model_names[i])

# Tüm tahminlerin birleşik regresyon grafiği
plt.figure(figsize=(10, 8))
for i, predict in enumerate(predicts):
    plt.scatter(y_test, predict, label=model_names[i])

plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "k--", lw=2)
plt.title("Regresyon Modellerinin Birleşik Tahminleri")
plt.xlabel("Gerçek Değerler")
plt.ylabel("Tahmin Edilen Değerler")
plt.legend()
plt.show()
