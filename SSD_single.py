#Librerias necesarias (OpenCV 4.0 y numpy)
import cv2 
import numpy as np 

path='/home/marco/Pruebas_FaceDetection/' #Se debe cambiar al path en el que se ubiquen las carpetas

#Inicializacion CNN tipo SSD
net = cv2.dnn.readNetFromCaffe("deploy.prototxt.txt","res10_300x300_ssd_iter_140000.caffemodel")


img= cv2.imread(path+'/original_pic/2003/01/01/big/img_336.jpg',1)

		
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
img_output = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)


#CNN->Etapa de deteccion

(h, w) = img_output.shape[:2]

#Normalizacion y resizing		
blob = cv2.dnn.blobFromImage(cv2.resize(img_output, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
#Entrada de la imagen a la red
net.setInput(blob)
#Obtencion de detecciones
detections = net.forward()

#Inicializacion lista de detecciones
list_det=[]

		
#Generacion de bounding boxes
for i in range(0, detections.shape[2]):

	confidence = detections[0, 0, i, 2]

			
	if confidence < 0.6:
		continue

						
	box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])

	(startX, startY, endX, endY) = box.astype("int")
	boxes_list=[startX, startY, endX-startX, endY-startY, confidence]

	#Dibujo de bounding boxes
	text = "{:.2f}%".format(confidence * 100)
	y = startY - 10 if startY - 10 > 10 else startY + 10
	cv2.rectangle(img, (startX, startY), (endX, endY), (0, 0, 255), 2)
	cv2.putText(img, text, (startX, y),
	cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
			

	cv2.imwrite("Image",img)
		#cv2.waitKey(0) 

