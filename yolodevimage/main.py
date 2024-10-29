import requests
from bs4 import BeautifulSoup
import os
import torch
import cv2

# YOLO modelini yükle
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Görselleri kaydetmek için klasör adı
image_folder = 'news_images'
if not os.path.exists(image_folder):  # Eğer klasör yoksa oluştur
    os.makedirs(image_folder)

# Farklı haber sitelerinin URL'leri
news_urls = [
    "https://www.bbc.com/news",
    "https://www.cnn.com",
    "https://www.reuters.com",
    "https://www.nytimes.com",
    "https://www.aljazeera.com",
    "https://www.foxnews.com",
    "https://www.nbcnews.com",
    "https://www.theguardian.com/international",
    "https://www.wsj.com",
    "https://www.cbsnews.com"
]


# Haber sitelerinden görselleri indirme
def fetch_images(url, folder, limit=10):
    # Haber sitesinden içerik çek
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')

    # İlk limit kadar img etiketini al
    images = soup.find_all('img')[:limit]

    # Her görsel için işlemleri yap
    for idx, img in enumerate(images):
        img_url = img.get('src')

        # Eğer img URL geçerli bir http bağlantısı ise
        if img_url and img_url.startswith('http'):
            try:
                # Görseli indir
                img_data = requests.get(img_url).content

                # Görsel dosyasını kaydet
                file_path = os.path.join(folder, f"image_{url.split('//')[1].split('.')[0]}_{idx}.jpg")
                with open(file_path, 'wb') as img_file:
                    img_file.write(img_data)
                print(f"İndirilen görsel: {img_url}")
            except Exception as e:
                print(f"Görsel indirilemedi: {img_url}, hata: {e}")


# Tüm sitelerden görselleri indir
for news_url in news_urls:
    fetch_images(news_url, image_folder)


# Görüntüleri sınıflandırma ve kategorilere ayırma
def classify_images(folder):
    # Sınıflandırma için kategori eşleştirmeleri
    categories = {
        'person': 'insan',
        'dog': 'hayvan',
        'cat': 'hayvan',
        'car': 'eşya',
        'bus': 'eşya',
        'truck': 'eşya',
        # Diğer nesneleri ekleyin
    }

    # Klasördeki her görsel dosyası için
    for img_name in os.listdir(folder):
        img_path = os.path.join(folder, img_name)
        img = cv2.imread(img_path)

        # YOLO ile tahmin yap
        results = model(img)
        results.render()  # Görüntü üzerine nesne kutularını çiz

        # Sonuçları yazdır ve görselleştir
        print(f"\nGörüntü: {img_name} için sınıflandırma sonuçları:")
        for det in results.pred[0]:
            # Nesne sınıfını al
            class_name = model.names[int(det[-1])]
            # Kategoriye göre eşleştirme yap
            category = categories.get(class_name, 'Diğer')
            print(f"Nesne: {class_name}, Kategori: {category}")

        # Sonuçları göster
        cv2.imshow("YOLO Sınıflandırma", results.ims[0])  # İşlenmiş görseli göster
        cv2.waitKey(0)  # Görseli kapatmak için bir tuşa basın
        cv2.destroyAllWindows()


# İndirilen görselleri sınıflandır ve kategorize et
classify_images(image_folder)
