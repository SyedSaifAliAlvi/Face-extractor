import glob
import imageio
import cv2
from Extraction_Face import extract_faces
loadimages = []
for image_path in glob.glob(r"C:\Users\AssassinTiger\PycharmProjects\Object_detection\\*.png"):
    im = imageio.imread(image_path)
    loadimages.append(im)
frequency = 0
haar_cascade_face = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
for img in loadimages:
    frequency = frequency+1
    name = str(frequency) + '.png'
    faces= extract_faces(haar_cascade_face, img)
    extracted_face = cv2.cvtColor(faces, cv2.COLOR_BGR2RGB)
    cv2.imwrite(str(name),extracted_face)