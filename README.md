# Face-Detection
Face detection results tested  in FDDB benchmark. (http://vis-www.cs.umass.edu/fddb/index.html)

First Download images from FDDB benchmark and put into a folder named original_pics in the main directory (tamaraberg.com/faceDataset/originalPics.tar.gz):

Unzip all folders in the main directory

Before testing the code, you need to change the path in all programs.

List of files/folders and description:

Haar_single.py: Viola-Jones algorithm implementation for a single image
Haar_test.py: Viola-Jones algorithm implementation for all images in FDDB benchmark

SSD_single.py: DNN OpenCv (SSD) implementation for a single image
SSD_test.py: DNN OpenCv (SSD) implementation for all images in FDDB benchmark

How to evaluate:
Go to evaluation folder in terminal and run :

make 

Run Haar_single.py or SSD_test.py in the terminal (dont forget to change the path):

python SSD_test.py

A file named test2.text must be created in /image_result1 folder

Go to evaluation folder and run runEvaluate.pl:

./runEvaluate.pl

Follow instructions to compare your ROC curves




