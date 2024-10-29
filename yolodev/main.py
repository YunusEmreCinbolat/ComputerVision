import torch  # PyTorch kütüphanesini içe aktarıyoruz
import cv2  # OpenCV kütüphanesini içe aktarıyoruz (Görüntü işleme için)

# YOLOv5 modelini yükleyin
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)  # YOLOv5 küçük modelini (yolov5s) önceden eğitilmiş ağırlıklarla yüklüyoruz

# Nesne kategorileri
categories = {
    "İnsan": ["person"],  # "İnsan" kategorisi için sınıflar
    "Hayvan": ["cat", "dog", "bird", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe"],  # "Hayvan" kategorisi için sınıflar
    "Eşya": ["bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat", "traffic light",
             "fire hydrant", "stop sign", "parking meter", "bench", "backpack", "umbrella", "handbag",
             "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
             "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
             "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
             "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed", "diningtable",
             "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
             "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
             "hair drier", "toothbrush"]  # "Eşya" kategorisi için sınıflar
}

# Web kamerasını başlatın
cap = cv2.VideoCapture(0)  # 0 numaralı cihaz (genellikle varsayılan web kamerası) üzerinden video akışı başlatıyoruz

# Kamera açık olduğu sürece döngü
while cap.isOpened():
    ret, frame = cap.read()  # Kameradan bir kare okuyoruz
    if not ret:  # Eğer görüntü alınamadıysa
        print("Web kamerasından görüntü alınamadı.")
        break  # Döngüyü sonlandırıyoruz

    # Model ile nesne algılama
    results = model(frame)  # Kareyi YOLOv5 modeline vererek nesne algılama işlemini yapıyoruz

    # Algılanan nesneleri çerçeve üzerinde çiz
    for obj in results.pred[0]:  # Algılanan nesneleri döngüye alıyoruz
        class_id = int(obj[5])  # Sınıf kimliğini alıyoruz
        class_name = model.names[class_id]  # Nesnenin adını sınıf kimliğine göre alıyoruz
        x1, y1, x2, y2 = map(int, obj[:4])  # Nesnenin koordinatlarını alıyoruz

        # Nesne kategorisini belirleyin
        category = "Diğer"  # Varsayılan olarak kategoriyi "Diğer" yapıyoruz
        for key, values in categories.items():  # Kategorileri kontrol ediyoruz
            if class_name in values:  # Eğer nesne adı kategoride varsa
                category = key  # O kategoriye atıyoruz
                break  # Döngüden çıkıyoruz

        # Çerçeve üzerine sınıf adı ve kategori yazdırın
        label = f"{class_name} ({category})"  # Etiket olarak nesne adı ve kategori yazıyoruz
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Nesne etrafında yeşil bir dikdörtgen çiziyoruz
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)  # Dikdörtgenin üstüne nesne adını yazıyoruz

    # Çerçeveyi göster
    cv2.imshow("YOLOv5 Nesne Algılama", frame)  # Güncellenmiş kareyi ekranda gösteriyoruz

    # 'q' tuşuna basarak çıkın
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Eğer 'q' tuşuna basılırsa
        break  # Döngüyü sonlandırıyoruz

# Temizleme işlemleri
cap.release()  # Kamerayı serbest bırakıyoruz
cv2.destroyAllWindows()  # Tüm OpenCV pencerelerini kapatıyoruz
