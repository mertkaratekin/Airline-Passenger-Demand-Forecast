**Havayolu Yolcu Talep Tahmini Projesi**
Bu proje, havayolu yolcu talep tahminini regresyon modelleri ve yapay sinir ağı yöntemleri kullanarak gerçekleştiren bir Python uygulamasını içermektedir. Proje, veri analizi, model eğitimi ve performans değerlendirmesi adımlarını içermektedir.

**Kullanılan Teknolojiler ve Kütüphaneler**
Python: Proje, Python programlama dilinde yazılmıştır.
NumPy: Sayısal hesaplamalar ve veri manipülasyonu için kullanılmıştır.
Pandas: Veri analizi ve işlemleri için kullanılmıştır.
Matplotlib ve Seaborn: Veri görselleştirme işlemleri için kullanılmıştır.
scikit-learn (sklearn): Makine öğrenimi algoritmaları ve araçları sağlayan bir kütüphane olarak kullanılmıştır.
MLPRegressor (Yapay Sinir Ağı Regresyon Modeli): Yapay sinir ağı tabanlı regresyon modeli oluşturmak için kullanılmıştır.
**Kullanım**
Projenin ana bileşeni olan HavayoluYolcuTahmini.py dosyasını çalıştırarak proje başlatılabilir. Bu dosya, veri setini yükler, modeli eğitir, tahminlerde bulunur ve sonuçları değerlendirir.

**Proje Aşamaları**
Veri Setinin Yüklenmesi: HavayoluYolcu.csv dosyasından veri seti okunur.
Veri Ön İşleme: Bağımlı değişken ve bağımsız değişkenler ayrılır. Veri seti eğitim ve test kümelerine bölünür. Veri normalizasyonu yapılır.
Model Eğitimi: Farklı regresyon modelleri ve yapay sinir ağı modeli eğitilir.
Performans Değerlendirmesi: Her bir modelin tahmin performansı değerlendirilir ve metrikler hesaplanır.
Sonuçların Görselleştirilmesi: Tahminler ve gerçek değerler arasındaki ilişki grafiklerle gösterilir.


**Model Performansı**
Modellerin performansı aşağıdaki metriklerle değerlendirilmiştir:

Mean Absolute Error (MAE)
Mean Squared Error (MSE)
Root Mean Squared Error (RMSE)
R2 Score

**Sonuçlar**
Projede, yapılan tahminlerin gerçek verilere olan yakınlığı ve modellerin performansı değerlendirilmiştir. Ayrıca, regresyon modellerinin tahminlerini gösteren grafikler oluşturulmuştur.
