import cv2
import numpy

kamera = cv2.VideoCapture(0)

while (True):
    ret, VideoImage = kamera.read()

    scale_percent = 50  # %50 küçültme oranı
    width = int(VideoImage.shape[1] * scale_percent / 100)
    height = int(VideoImage.shape[0] * scale_percent / 100)
    dim = (width, height)

    # Orijinal görüntüyü yeniden boyutlandır
    resized_VideoImage = cv2.resize(VideoImage, dim, interpolation=cv2.INTER_AREA)

    median_filter_video=cv2.medianBlur(resized_VideoImage,5)
    resized_median_filtered = cv2.resize(median_filter_video, dim, interpolation=cv2.INTER_AREA)

    laplaced_filter_video= cv2.Laplacian(median_filter_video,cv2.CV_64F)
    laplaced_filter_video =numpy.uint8(numpy.absolute(laplaced_filter_video))
    resized_laplace_filtered = cv2.resize(laplaced_filter_video, dim, interpolation=cv2.INTER_AREA)

    frames=numpy.hstack((resized_VideoImage,resized_median_filtered,resized_laplace_filtered))

    cv2.imshow("Orijinal | Median Filtre | Laplace Filtre", frames)
    if cv2.waitKey(50) & 0xFF == 27 :
        break

kamera.release()
cv2.destroyAllWindows()