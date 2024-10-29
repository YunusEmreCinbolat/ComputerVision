# Gerekli kütüphaneleri içe aktarıyoruz
import cv2  # Görüntü işleme işlemleri için OpenCV kütüphanesi
import matplotlib.pyplot as plt  # Görüntüyü grafiksel olarak göstermek için Matplotlib

# Görüntü dosyasının yolunu belirliyoruz
image_path = 'C:\\Users\Dell\Documents\\4.sinif\ComputerVision\codes\BorderRecognition\img.png'

# Görüntüyü gri tonlamalı olarak yüklüyoruz
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Canny algoritması ile görüntüdeki kenarları tespit ediyoruz
edges = cv2.Canny(image, threshold1=50, threshold2=150)  # 50 ve 150, kenar hassasiyeti için eşik değerleri

# Görüntüyü göstermek için bir figür oluşturuyoruz ve boyutunu ayarlıyoruz
plt.figure(figsize=(6, 6))

# Kenarları içeren görüntüyü çiziyoruz; `cmap='gray'` ile gri tonlamalı olarak gösteriyoruz
plt.imshow(edges, cmap='gray')

# Görüntü üzerindeki eksenleri gizliyoruz
plt.axis('off')

# Kenar tespit edilmiş görüntüyü gösteriyoruz
plt.show()
