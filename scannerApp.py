# import the necessary packages
from skimage.filters import threshold_local
import numpy as np
import argparse
import cv2
import imutils
from datetime import datetime

def order_points(pts):

	rect = np.zeros((4, 2), dtype="float32")

	s = pts.sum(axis=1)
	rect[0] = pts[np.argmin(s)]
	rect[2] = pts[np.argmax(s)]

	diff = np.diff(pts, axis=1)
	rect[1] = pts[np.argmin(diff)]
	rect[3] = pts[np.argmax(diff)]

	return rect


def transformFourPoints(image, pts):

	rect = order_points(pts)
	(tl, tr, br, bl) = rect

	widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
	widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
	maxWidth = max(int(widthA), int(widthB))

	heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
	heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
	maxHeight = max(int(heightA), int(heightB))

	dst = np.array([[0, 0],	[maxWidth - 1, 0],	[maxWidth - 1, maxHeight - 1],	[0, maxHeight - 1]], dtype="float32")

	M = cv2.getPerspectiveTransform(rect, dst)
	warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

	return warped


ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True, help = "Path to the image to be scanned")
args = vars(ap.parse_args())

image = cv2.imread(args["image"])
ratio = image.shape[0] / 500.0
orig = image.copy()
image = imutils.resize(image, height = 500)


gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (5, 5), 0)
edged = cv2.Canny(gray, 75, 200)


print("STEP 1: Edge Detection")
cv2.imshow("Image", image)
cv2.imshow("Edged", edged)
cv2.waitKey(1000)  # waits 1 second, then continues
cv2.destroyAllWindows()
###
cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]

if not cnts:
	print("No contours found.")
	exit(1)

try:
	cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]
except cv2.error as e:
	print("Error while sorting contours:", e)
	exit(1)

for c in cnts:

	peri = cv2.arcLength(c, True)
	approx = cv2.approxPolyDP(c, 0.02 * peri, True)

	if len(approx) == 4:
		screenCnt = approx
		break

print("STEP 2: Finding contours of paper")
cv2.drawContours(image, [screenCnt], -1, (0, 255, 0), 2)
cv2.imshow("Outline", image)
cv2.waitKey(1000)  # waits 1 second, then continues
cv2.destroyAllWindows()

warped = transformFourPoints(orig, screenCnt.reshape(4, 2) * ratio)

# Load pre-trained DNN face detector
net = cv2.dnn.readNetFromCaffe("deploy.prototxt", "res10_300x300_ssd_iter_140000.caffemodel")
# Resize image to 300x300 and convert to blob
blob = cv2.dnn.blobFromImage(warped, 1.0, (300, 300), (104.0, 177.0, 123.0), swapRB=True, crop=False)
net.setInput(blob)


# === FACE DETECTION BEFORE GRAYSCALE ===
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Use the color image (BGR)
faces = face_cascade.detectMultiScale(warped, scaleFactor=1.1, minNeighbors=40)

if len(faces) == 0:
	print("❌ No face detected.")
else:
	print(f"✅ Detected {len(faces)} face(s).")
	for i, (x, y, w, h) in enumerate(faces):
		# ➕ Add padding (e.g., 20 pixels around each side)
		pad = 40
		x1 = max(x - pad, 0)
		y1 = max(y - pad, 0)
		x2 = min(x + w + pad, warped.shape[1])
		y2 = min(y + h + pad, warped.shape[0])

		# Crop with padding
		face_img = warped[y1:y2, x1:x2]

		# Optional: upscale for saving
		upscale_factor = 2  # or 1.5, etc.
		face_img = cv2.resize(face_img, None, fx=upscale_factor, fy=upscale_factor, interpolation=cv2.INTER_CUBIC)

		# Save the larger face image
		face_filename = f"face_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_{i + 1}.png"
		cv2.imwrite(face_filename, face_img)
		print(f"💾 Saved enlarged face image as {face_filename}")

		# Draw rectangle on original image
		cv2.rectangle(warped, (x1, y1), (x2, y2), (0, 255, 0), 2)

		# Show enlarged face
		cv2.imshow(f"Face {i + 1}", face_img)
		cv2.waitKey(1000)
		cv2.destroyAllWindows()

offset_degree = int(input("Enter degree of your noise(recommend between 15 - 32):"))
warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
T = threshold_local(warped, 11, offset=offset_degree, method="gaussian")


#add treshhold 0-1!!!!!!!!!!!!




warped = (warped > T).astype("uint8") * 255


print("STEP 3: Applying perspective transform")
cv2.imshow("Original", imutils.resize(orig, height=650))
cv2.imshow("Scanned", imutils.resize(warped, height=650))
cv2.waitKey(5000)


# Get the current time and format it as a string (e.g., "2025-05-03_14-30-25")
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# You can save the original image or any other image you have (e.g., warped_binary, etc.)
filename = f"scanned_image_{timestamp}.png"  # File name with timestamp

# Save the image to disk
cv2.imwrite(filename, warped)  # Save 'warped' image with timestamped name
print(f"Image saved as {filename}")
