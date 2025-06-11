import numpy as np
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt

# Veriyi yükle
(X_train, y_train), _ = mnist.load_data()

# MNIST verisi X_train içinde numpy array olarak 28x28 boyutlu gri tonlamalı görüntüler içerir.
# 0 → siyah   255 → beyaz   Aradaki değerler → gri tonları
# / 255.0 ile normalizasyon yapılır, yani tüm değerler 0.0–1.0 aralığına çekilir.

X = X_train[:1] / 255.0
y = y_train[:1]

print("Girdi görüntüsü boyutu:", X.shape)
print("Girdi görüntüsü boyutu:", len(X))


def conv2d(input_img, filt, bias, stride=1):
    h, w = input_img.shape
    f_h, f_w = filt.shape
    out_h = (h - f_h) // stride + 1
    out_w = (w - f_w) // stride + 1
    output = np.zeros((out_h, out_w))

    for i in range(out_h):
        for j in range(out_w):
            region = input_img[i:i+f_h, j:j+f_w]
            output[i, j] = np.sum(region * filt) + bias
    return output


np.random.seed(42)
filter_ = np.random.randn(3, 3)
bias = np.random.randn()


img = X[0]  # 28x28
conv_output = conv2d(img, filter_, bias)
print("Conv2D çıktısı boyutu:", conv_output.shape)
print("Conv2D çıktısı:\n", conv_output)


def relu(x):
    return np.maximum(0, x)

relu_output = relu(conv_output)
print("ReLU çıktısı:\n", relu_output)



def maxpool2d(input_map, size=2, stride=2):
    h, w = input_map.shape
    out_h = (h - size) // stride + 1
    out_w = (w - size) // stride + 1
    output = np.zeros((out_h, out_w))

    for i in range(out_h):
        for j in range(out_w):
            region = input_map[i*stride:i*stride+size, j*stride:j*stride+size]
            output[i, j] = np.max(region)
    return output

pooled_output = maxpool2d(relu_output)
print("MaxPooling çıktısı boyutu:", pooled_output.shape)
print("MaxPooling çıktısı:\n", pooled_output)


flattened = pooled_output.flatten()
print("Flattened çıktı boyutu:", flattened.shape)
print("Flattened vektör:\n", flattened)


W_dense = np.random.randn(10, flattened.shape[0]) * 0.01
b_dense = np.random.randn(10)

z = np.dot(W_dense, flattened) + b_dense



def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / np.sum(e_x)

output_probs = softmax(z)
print("Softmax çıktı (sınıf olasılıkları):\n", output_probs)
print("Tahmin edilen sınıf:", np.argmax(output_probs))




