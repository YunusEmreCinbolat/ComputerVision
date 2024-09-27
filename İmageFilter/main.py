import cv2
import matplotlib
import matplotlib.pyplot as plt

image = cv2.imread('city.jpeg')

im_rgb = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)


median_filter_image = cv2.blur(im_rgb,(4,4))


laplace_filtered_image = cv2.Laplacian(median_filter_image,cv2.CV_64F)


plt.figure(figsize=(15,5))

plt.subplot(1,3,1)
plt.imshow(image)
plt.title("Orijinal Resim")
plt.axis("off")

plt.subplot(1,3,2)
plt.imshow(median_filter_image)
plt.title("Orijinal Resme Medyan Filresi Uygulanmis Resim")
plt.axis("off")

plt.subplot(1,3,3)
plt.imshow(laplace_filtered_image)
plt.title("Medyan filtresi Uygulanmis Resme Laplas Filtresi Uygulanmis Hali")
plt.axis("off")


plt.show()