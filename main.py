import cv2
import pytesseract
from pytesseract import Output

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe'

img = cv2.imread('target.jpg')
config = ' --psm 3 outputbase digits'
data = pytesseract.image_to_string(img, lang='eng', config=config)
print(data)

#show
height = img.shape[0]
width = img.shape[1]
d = pytesseract.image_to_boxes(img, output_type=Output.DICT, config=config)
n_boxes = len(d['char'])
for i in range(n_boxes):
    (text, x1, y2, x2, y1) = (d['char'][i], d['left'][i], d['top'][i], d['right'][i], d['bottom'][i])
    cv2.rectangle(img, (x1, height - y1), (x2, height - y2), (0, 128, 128), 2)
cv2.imshow('img', img)
cv2.waitKey(0)
