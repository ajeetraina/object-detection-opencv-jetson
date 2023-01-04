# Object detection using deep learning with OpenCV and Python 

OpenCV `dnn` module supports running inference on pre-trained deep learning models from popular frameworks like Caffe, Torch and TensorFlow. 

When it comes to object detection, popular detection frameworks are
 * YOLO
 * SSD
 * Faster R-CNN
 
 Support for running YOLO/DarkNet has been added to OpenCV dnn module recently. 
 
 ## Dependencies
  * opencv
  * numpy
  
`pip install numpy opencv-python`

**Note: Compatability with Python 2.x is not officially tested.**

 ## YOLO (You Only Look Once)
 
 Download the pre-trained YOLO v3 weights file from this [link](https://pjreddie.com/media/files/yolov3.weights) and place it in the current directory or you can directly download to the current directory in terminal using
 
 `$ wget https://pjreddie.com/media/files/yolov3.weights`
 
 Provided all the files are in the current directory, below command will apply object detection on the input image `dog.jpg`.
 
 `$ python yolo_opencv.py --image dog.jpg --config yolov3.cfg --weights yolov3.weights --classes yolov3.txt`
 
 
 **Command format** 
 
 _$ python yolo_opencv.py --image /path/to/input/image --config /path/to/config/file --weights /path/to/weights/file --classes /path/to/classes/file_
 
 Checkout the [blog post](http://www.arunponnusamy.com/yolo-object-detection-opencv-python.html) to learn more.
 
 ### sample output :
 ![](object-detection.jpg)
 
Checkout the object detection implementation available in [cvlib](http:cvlib.net) which enables detecting common objects in the context through a single function call `detect_common_objects()`.
 
 
 (_SSD and Faster R-CNN examples will be added soon_)
 
 
 ## Face Detection 
 
 ### Running 01_face_dataset.py
 
 The first script is called 01_face_dataset.py. We will use this python script to capture your face and store it as a dataset.
 Once you run the below python script, it will activate your camera, capture your face image and then store it as dataset.
 
 ```
 python3 01_face_dataset.py
[ WARN:0] global /home/nvidia/host/build_opencv/nv_opencv/modules/videoio/src/cap_gstreamer.cpp (933) open OpenCV | GStreamer warning: Cannot query video position: status=0, value=-1, duration=-1

 Enter user ID end press <Enter> ==>  ajeetraina

 Initializing face capture. Look the camera and wait ...

 [INFO] Exiting Program and cleanup stuff
 ```

### Running 02_face_training.py

You will need `pillow` Python module to get this script executed.
Let's install the following

```
pip3 install pillow
```

Now, run the script:

```
python3 02_face_training.py
```


### Running 03_face_recognition.py

Finally, you should be able to do facial recognition via this script

```
python3 03_face_recognition.py
```
