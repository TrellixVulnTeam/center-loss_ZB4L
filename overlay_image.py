# overlay images
import cv2

background = cv2.imread("images/good1.png")
overlay = cv2.imread("images/screw.png")

print(f"background size : {background.shape} | overlay size : {overlay.shape}")

### Rescale overlay

scale_percent = 10
width = int(overlay.shape[1] * scale_percent / 100)
height = int(overlay.shape[0] * scale_percent / 100)
dim = (width, height)

resized = cv2.resize(overlay, dim, interpolation=cv2.INTER_AREA)

print("Resized Dimensions : ", resized.shape)

cv2.imshow("Resized image", resized)
cv2.waitKey(0)
cv2.destroyAllWindows()

### Make overlay with background

added_image = cv2.addWeighted(background, 0.4, resized, 0.1, 0)

cv2.imshow("Overlay", added_image)
cv2.waitKey(0)
cv2.destroyAllWindows()