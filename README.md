
## 🚀 Proje Hakkında

Kod, derin öğrenme kütüphanelerine bağımlı kalmadan, temel matris işlemleriyle bir CNN'in nasıl çalıştığını göstermeyi amaçlamaktadır. Eğitim verisi olarak TensorFlow/Keras aracılığıyla MNIST veri kümesi kullanılır, ancak modelin tüm öğrenme mantığı elle yazılmıştır.

Ana bileşenler:

* **Veri Yükleme**: TensorFlow ile MNIST verisi alınır.
* **Katmanlar**:

  * Convolution
  * ReLU aktivasyonu
  * MaxPooling
  * Flatten (düzleştirme)
  * Dense (tam bağlantılı katman)
  * Softmax (çıktı)
* **İleri Yayılım (Forward Propagation)**
* **Geri Yayılım (Backpropagation)** — basitleştirilmiş biçimde

## 📂 Dosya Yapısı

```bash
├── Cnn.py       # Tüm ağ mimarisi ve eğitim işlemi
```

## 🔧 Gereksinimler

* Python 3.8+
* NumPy
* TensorFlow (sadece veri yüklemek için kullanılır)

Kurmak için:

```bash
pip install numpy tensorflow
```

## 🏃‍♀️ Nasıl Çalıştırılır?

```bash
python Cnn.py
```

İlk çalıştırmada MNIST verisi indirilecek ve model eğitilecektir. Eğitim çıktısı konsola yazdırılır.

