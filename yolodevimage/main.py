import requests
from bs4 import BeautifulSoup
import os
import torch
import cv2
import numpy as np

# Haber sitesinden görüntüleri çekme fonksiyonu
def fetch_images_from_news_page(url, save_folder):
    # Sayfayı indir
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')

    # Görselleri bul ve indir
    images = soup.find_all('img')
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    for idx, img in enumerate(images):
        img_url = img.get('src')
        # Sadece tam URL’li görselleri indir
        if img_url and img_url.startswith('http'):
            try:
                img_data = requests.get(img_url).content
                with open(os.path.join(save_folder, f"image_{idx}.jpg"), 'wb') as img_file:
                    img_file.write(img_data)
                print(f"İndirilen görsel: {img_url}")
            except Exception as e:
                print(f"Görsel indirilemedi: {img_url}, hata: {e}")

# YOLO modelini başlat
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Görüntüleri sınıflandırma ve kategorilere ayırma fonksiyonu
def classify_and_categorize_images(image_folder):
    categories = {
        'person': 'insan',
        'dog': 'hayvan',
        'cat': 'hayvan',
        'car': 'eşya',
        'bus': 'eşya',
        'truck': 'eşya',
        # İhtiyaca göre diğer nesneleri ekleyin
    }

    for img_name in os.listdir(image_folder):
        img_path = os.path.join(image_folder, img_name)
        img = cv2.imread(img_path)

        try:
            # YOLO ile tahmin yap
            results = model(img)
        except Exception as e:
            print(f"Model çalışırken hata oluştu: {e}")
            continue

        # Sonuçları yazdır ve kategorilere ayır
        print(f"\nGörüntü: {img_name} için sınıflandırma sonuçları:")
        for det in results.pred[0]:
            class_name = model.names[int(det[-1])]
            category = categories.get(class_name, 'Diğer')
            print(f"Nesne: {class_name}, Kategori: {category}")

        # Algılanan nesneleri ve kutucukları görselleştir
        results.render()  # Bu, görüntüyü kutucuklarla işaretler

        # Rendered image'ı OpenCV ile göster
        for img_rendered in results.imgs:  # Rendered sonuçları al
            img_show = np.squeeze(img_rendered)  # İlk boyutu atla
            cv2.imwrite('output_image.jpg', img_show)  # Görüntüyü kaydet
            cv2.imshow("YOLO Sınıflandırma", img_show)
            cv2.waitKey(0)  # Bir tuşa basılmasını bekleyin, ardından sıradaki görüntüye geçin

    cv2.destroyAllWindows()  # Tüm pencereleri kapat

# Kullanım
# Örnek bir haber sayfası URL'si kullanarak görselleri indir
fetch_images_from_news_page('https://image.cnnturk.com/i/cnnturk/75/0x186/671fa1a8917176d0da28b060.jpg', 'news_images2')

# İndirilen görselleri sınıflandır ve kategorize et
classify_and_categorize_images('news_images2')
