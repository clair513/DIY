# Importing required packages:
from PIL import Image
import pytesseract
import argparse
import cv2
import os

# Constructing argument parser:
ap = argparse.ArgumentParser(add_help=False)
ap.add_argument('-i', '--image', required=True)
ap.add_argument('-p', '--preprocess', type=str, default='thresh')
args = vars(ap.parse_args())

# Loading image & binarizing(grayscale) it:
img = cv2.imread(args['image'])
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Verifying application of 'threshold' for preprocessing image [Otsu's Binarization]:
if args['preprocess'] == 'thresh':
	gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
# Verifying if Median blurring is required for Noise reduction [For Salt & Pepper background]:
elif args['preprocess'] == 'blur':
	gray = cv2.medianBlur(gray, 3)

# Writing grayscale image to disk (as a temporary file) to apply OCR on it:
filename = '{}.png'.format(os.getpid())
cv2.imwrite(filename, gray)

# Loading image (as a PIL/Pillow image), OCR, and then deleting temp file:
text = pytesseract.image_to_string(Image.open(filename), lang='eng')
os.remove(filename)

# Saving parsed Output:
with open('pdf_output.txt', 'a') as f:
	f.write(text)

#USAGE:
#python ocr.py --image images/input_walmart.jpg
