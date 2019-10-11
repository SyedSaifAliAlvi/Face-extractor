import cv2
def converttoRGB(image):
    return cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
haar_cascade_face = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
def extract_faces(cascade,test_image,scaleFactor = 1.1):
    image_copy = test_image.copy()
    gray_image = cv2.cvtColor(image_copy,cv2.COLOR_BGR2GRAY)
    faces_rect = cascade.detectMultiScale(gray_image,scaleFactor=scaleFactor, minNeighbors = 5)
    imagec=None
    for (x,y,w,h) in faces_rect:
        imagec = image_copy[y:y+h,x:x+w]
    return imagec