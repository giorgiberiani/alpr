# ALPR localization and recognition

The project is split into several parts:
* Localize the car in the image and take the coordinates of the car.
* Localize the ALPR in the car image using image and coordinates of the car from previous step.
* Localize the characters in the ALPR image
* Recognize the characters from previous step


For the localizing the cars, we use pre trained model of YOLOv3-608 done on coco dataset. The model contains 80 classes
from which we use 4 classes: Cars, Motorbike, Bus, Truck. Model is the most precise one and can be done only on 20 fps.

The same model is used for localizing the ALPR. We trained it on out dataset. Firstly we took car images from an input image and than get the bounding boxes of ALPR from car's image and trained the model on that information:
car image + bounding box of ALPR. As we wanted the maximum precision the input image was increased to 608 x 608 as it is the maximum input for YOLO. 

The models can be found <a href='https://drive.google.com/open?id=1i4wW_d4oZDp-icTOGNQ2SapdNFnU_2Ky'> here </a>
You should place them into the folder yolo/wights/ folder.

###  Running the program
Install the requirements with 'pip install -r requirements.txt'
Into the file alpr_detection.py change the value INPUT_IMAGE to the image path you want to run detection on.
Then run 'python alpr_detection.py'



