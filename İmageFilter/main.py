# Gerekli kütüphaneleri içe aktarıyoruz
import cv2  # Görüntü işleme işlemleri için OpenCV
import matplotlib.pyplot as plt  # Görüntüyü grafiksel olarak göstermek için Matplotlib

# Görüntüyü yüklüyoruz
image = cv2.imread('city.jpeg')  # 'city.jpeg' dosyasını yüklüyor

# BGR renk uzayından RGB'ye dönüştürüyoruz
im_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # OpenCV varsayılan olarak BGR formatında görüntü okur, bu yüzden RGB'ye dönüştürüyoruz

# Medyan filtre uyguluyoruz
median_filter_image = cv2.blur(im_rgb, (4,4))  # 4x4 boyutunda medyan filtre uygulayarak görüntüyü yumuşatıyoruz

# Laplace filtresi uyguluyoruz
laplace_filtered_image = cv2.Laplacian(median_filter_image, cv2.CV_64F)  # Kenar tespiti için Laplace filtresi uyguluyoruz

# Grafik için figür boyutunu ayarlıyoruz
plt.figure(figsize=(15,5))

# Orijinal resmi gösteriyoruz
plt.subplot(1, 3, 1)  # 1 satır, 3 sütunlu gridin ilk bölümü
plt.imshow(image)  # Orijinal resmi gösteriyoruz
plt.title("Orijinal Resim")  # Başlık ekliyoruz
plt.axis("off")  # Eksenleri gizliyoruz

# Medyan filtre uygulanmış resmi gösteriyoruz
plt.subplot(1, 3, 2)  # 1 satır, 3 sütunlu gridin ikinci bölümü
plt.imshow(median_filter_image)  # Medyan filtre uygulanmış resmi gösteriyoruz
plt.title("Orijinal Resme Medyan Filresi Uygulanmis Resim")  # Başlık ekliyoruz
plt.axis("off")  # Eksenleri gizliyoruz

# Laplace filtresi uygulanmış resmi gösteriyoruz
plt.subplot(1, 3, 3)  # 1 satır, 3 sütunlu gridin üçüncü bölümü
plt.imshow(laplace_filtered_image)  # Laplace filtresi uygulanmış resmi gösteriyoruz
plt.title("Medyan Filtresi Uygulanmis Resme Laplas Filtresi Uygulanmis Hali")  # Başlık ekliyoruz
plt.axis("off")  # Eksenleri gizliyoruz

# Grafiği gösteriyoruz
plt.show()  # Tüm görüntüleri tek bir figürde gösteriyoruz
