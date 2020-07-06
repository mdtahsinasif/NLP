#pip install opencv-python
#pip install tesseract
#pip install tesseract-ocr
import pytesseract
from PIL import Image
pytesseract.pytesseract.tesseract_cmd = r".....tesseract.exe"

img = Image.open("........\PhishingEmail1.png")
text = pytesseract.image_to_string(img)
print(text)
