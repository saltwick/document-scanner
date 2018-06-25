import cv2
import numpy as np
import argparse
import imutils
from cv_tools import four_point_transform
from skimage.filters import threshold_local

# Get image argument
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", help = "Path to image", required=True)

args = vars(ap.parse_args())

# Read specified image
img = cv2.imread(args["image"])
# Save a copy
orig = img.copy()

# Resize image
rat = img.shape[0] /500.0
img = imutils.resize(img, height = 500)

# Gray scale and blur for performance
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (5,5), 0)

# Detect edges using Canny Edge Detection
edge = cv2.Canny(gray, 75, 200)

# Use edges to find contours
cnts = cv2.findContours(edge.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if imutils.is_cv2() else cnts[1]
cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:5]
screenCnt = None

# Iterate through contours
for c in cnts:
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)
    # Identify rectangles (4 corners)
    if len(approx) == 4:
        screenCnt = approx
        break
# Use 4 point transform tool to perform a matrix transformation
warp = four_point_transform(orig, screenCnt.reshape(4,2) * rat)
warp = cv2.cvtColor(warp, cv2.COLOR_BGR2GRAY)
T = threshold_local(warp, 11, offset = 10, method = "gaussian")
warp = (warp > T).astype("uint8") * 255

# Display image and scanned version
while(True):
    cv2.imshow("Scanned", imutils.resize(warp, height = 650))
    cv2.imshow("original", imutils.resize(orig, height = 650))
    if cv2.waitKey(6) & 0xFF == ord('q'):
        break   
cv2.destroyAllWindows()

