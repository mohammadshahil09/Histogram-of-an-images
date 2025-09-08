# Ex 03: Histogram-of-an-images
## Aim
To obtain a histogram for finding the frequency of pixels in an Image with pixel values ranging from 0 to 255. Also write the code using OpenCV to perform histogram equalization.

## Software Required:
Anaconda - Python 3.7

## Algorithm:
### Step1:
Read the gray and color image using imread()

### Step2:
Print the image using imshow().



### Step3:
Use calcHist() function to mark the image in graph frequency for gray and color image.

### step4:
Use calcHist() function to mark the image in graph frequency for gray and color image.

### Step5:
The Histogram of gray scale image and color image is shown.


## Program:
```python
# Developed By: MOHAMMAD SHAHIL
# Register Number: 212223240044

import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read the image in grayscale format.
img = cv2.imread('Chennai_Central.jpg', cv2.IMREAD_GRAYSCALE)

# Display the images.
plt.imshow(img, cmap='gray')
plt.title('Original Image')
plt.show()

# Display the images
plt.hist(img.ravel(),256,range = [0, 256]);
plt.title('Original Image')
plt.show()

img_eq = cv2.equalizeHist(img)

# Display the images.
plt.hist(img_eq.ravel(), 256, range = [0, 256])
plt.title('Equalized Histogram')

# Display the images.
plt.imshow(img_eq, cmap='gray')
plt.title('Original Image')
plt.show()

# Read the color image.
img = cv2.imread('parrot.jpg', cv2.IMREAD_COLOR)

import cv2

img = cv2.imread('Chennai_Central.jpg')  # make sure extension is correct

if img is None:
    print("Image not loaded. Check path or filename.")
else:
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    print("Converted to HSV successfully.")

# Convert to HSV.
img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# Perform histogram equalization only on the V channel, for value intensity.
img_hsv[:,:,2] = cv2.equalizeHist(img_hsv[:, :, 2])

# Convert back to BGR format.
img_eq = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)
plt.imshow(img_eq[:,:,::-1]); plt.title('Equalized Image');plt.show()

plt.hist(img_eq.ravel(),256,range = [0, 256]); plt.title('Histogram Equalized');plt.show()

# Display the images.
#plt.figure(figsize = (20,10))
plt.subplot(221); plt.imshow(img[:, :, ::-1]); plt.title('Original Color Image')
plt.subplot(222); plt.imshow(img_eq[:, :, ::-1]); plt.title('Equalized Image')
plt.subplot(223); plt.hist(img.ravel(),256,range = [0, 256]); plt.title('Original Image')
plt.subplot(224); plt.hist(img_eq.ravel(),256,range = [0, 256]); plt.title('Histogram Equalized');plt.show()

# Display the histograms.
plt.figure(figsize = [15,4])
plt.subplot(121); plt.hist(img.ravel(),256,range = [0, 256]); plt.title('Original Image')
plt.subplot(122); plt.hist(img_eq.ravel(),256,range = [0, 256]); plt.title('Histogram Equalized')
```

## Output:
<img width="765" height="515" alt="image" src="https://github.com/user-attachments/assets/e03c4e6b-c893-4ef0-838c-221b9bd1f74d" />

<img width="757" height="567" alt="image" src="https://github.com/user-attachments/assets/21f57b4f-f682-43b8-97db-16e0036da7f1" />

<img width="821" height="543" alt="image" src="https://github.com/user-attachments/assets/ee929fe6-f4d2-4167-8312-32487aba51b1" />

<img width="834" height="526" alt="image" src="https://github.com/user-attachments/assets/1561e255-2052-48f3-8df5-601fe49f2a6a" />

<img width="742" height="502" alt="image" src="https://github.com/user-attachments/assets/accb40ea-4a79-4b59-ad54-e2bd91ad7677" />

<img width="786" height="552" alt="image" src="https://github.com/user-attachments/assets/3f02b3a5-1537-4415-ae14-5d98b98c538b" />

<img width="801" height="549" alt="image" src="https://github.com/user-attachments/assets/0cc9bf8b-aa7d-40af-8b71-c364674d311c" />

<img width="1371" height="449" alt="image" src="https://github.com/user-attachments/assets/53629410-91b5-4366-ae0b-c31348848d92" />


## Result: 
Thus the histogram for finding the frequency of pixels in an image with pixel values ranging from 0 to 255 is obtained. Also,histogram equalization is done for the gray scale image using OpenCV.
