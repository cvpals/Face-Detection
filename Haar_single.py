#Librerias necesarias (OpenCV 4.0 y numpy)
import numpy as np
import cv2

#Inicializacion Clasificador Viola-Jones  
face_cascade = cv2.CascadeClassifier('haarcascade_frontalcatface_extended.xml')



#Funcion para corregir bounding box superpuestas 
def non_max_suppression_fast(boxes, overlapThresh):

	if len(boxes) == 0:
		return []

	if boxes.dtype.kind == "i":
		boxes = boxes.astype("float")
 
	pick = []
 

	x1 = boxes[:,0]
	y1 = boxes[:,1]
	x2 = boxes[:,2]
	y2 = boxes[:,3]

	area = (x2 - x1 + 1) * (y2 - y1 + 1)
	idxs = np.argsort(y2)

	while len(idxs) > 0:
		last = len(idxs) - 1
		i = idxs[last]
		pick.append(i)
 

		xx1 = np.maximum(x1[i], x1[idxs[:last]])
		yy1 = np.maximum(y1[i], y1[idxs[:last]])
		xx2 = np.minimum(x2[i], x2[idxs[:last]])
		yy2 = np.minimum(y2[i], y2[idxs[:last]])
 

		w = np.maximum(0, xx2 - xx1 + 1)
		h = np.maximum(0, yy2 - yy1 + 1)
 

		overlap = (w * h) / area[idxs[:last]]
 

		idxs = np.delete(idxs, np.concatenate(([last],
			np.where(overlap > overlapThresh)[0])))
 

	return boxes[pick].astype("int")



#Lectura de imagen
img = cv2.imread('12_Group_Team_Organized_Group_12_Group_Team_Organized_Group_12_39.jpg') 

#Preprocesamiento
#cambio de imagen al canal YUV
img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
#Equalizacion canal de luminancia 
img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

#Ajuste de contraste		
lab= cv2.cvtColor(img_output, cv2.COLOR_BGR2LAB)

#Division en canales LAB
l, a, b = cv2.split(lab)

#CLAHE L-channel
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
cl = clahe.apply(l)

#Union de canales
limg = cv2.merge((cl,a,b))


#Retorno a RGB
img_output = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR


#Transformación GRAY
gray = cv2.cvtColor(img_output, cv2.COLOR_BGR2GRAY)

#Deteccion en el clasificador en cascada
faces = face_cascade.detectMultiScale(gray, 1.05, 1)
#Eliminacion de bounding boxes superpuestos
faces=non_max_suppression_fast(faces, 0.45)

#Generacion de bounding boxes 
for (x,y,w,h) in faces:
	
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)

cv2.imshow('img',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
