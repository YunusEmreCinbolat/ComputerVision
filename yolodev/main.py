import torch
import cv2

# YOLOv5 modelini yükleyin
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Nesne kategorileri
categories = {
    "İnsan": ["person"],
    "Hayvan": ["cat", "dog", "bird", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe"],
    "Eşya": ["bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat", "traffic light",
             "fire hydrant", "stop sign", "parking meter", "bench", "backpack", "umbrella", "handbag",
             "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
             "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
             "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
             "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed", "diningtable",
             "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
             "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
             "hair drier", "toothbrush"]
}

# Web kamerasını başlatın
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Web kamerasından görüntü alınamadı.")
        break

    # Model ile nesne algılama
    results = model(frame)

    # Algılanan nesneleri çerçeve üzerinde çiz
    for obj in results.pred[0]:
        class_id = int(obj[5])  # Sınıf kimliği
        class_name = model.names[class_id]  # Nesnenin adı
        x1, y1, x2, y2 = map(int, obj[:4])  # Nesnenin konumu

        # Nesne kategorisini belirleyin
        category = "Diğer"
        for key, values in categories.items():
            if class_name in values:
                category = key
                break

        # Çerçeve üzerine sınıf adı ve kategori yazdırın
        label = f"{class_name} ({category})"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Çerçeveyi göster
    cv2.imshow("YOLOv5 Nesne Algılama", frame)

    # 'q' tuşuna basarak çıkın
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Temizleme işlemleri
cap.release()
cv2.destroyAllWindows()
