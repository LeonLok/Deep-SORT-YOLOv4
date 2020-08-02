# Introduction
This project was inspired by:
* https://github.com/nwojke/deep_sort
* https://github.com/Ma-Dan/keras-yolo4
* https://github.com/Qidian213/deep_sort_yolov3

I swapped out YOLO v3 for YOLO v4 and added the option for asynchronous processing, which significantly improves
the FPS. However, FPS monitoring is disabled when asynchronous processing is used since it isn't accurate.

In addition, I took the algorithm from this [paper](https://ieeexplore.ieee.org/document/8909903) and implemented it into `deep_sort/track.py`.
The original method for confirming tracks was based simply on the number of times an object has been detected without considering detection confidence, leading to high tracking false positive rates when unreliable detections occur (i.e. low confidence true positives or high confidence false positives). The track filtering algorithm reduces this significantly by calculating the average detection confidence over a set number of detections before confirming a track.

See the comparison video below.

## Low confidence track filtering
[![Comparison Video Link](https://img.youtube.com/vi/01Geud6GmB4/0.jpg)](https://youtu.be/01Geud6GmB4)

Navigate to the appropriate folder to use low confidence track filtering. The above video demonstrates the difference.

See the settings section for parameter instructions.

## YOLO v3 and YOLO v4 comparison video with Deep SORT
[![Comparison Video Link](https://img.youtube.com/vi/_8WkO3hVOlY/0.jpg)](https://youtu.be/_8WkO3hVOlY)

## With asynchronous processing
![](gifs/async_example.gif)

As you can see in the gif, asynchronous processing has better FPS but causes stuttering.

This code only detects and tracks people, but can be changed to detect other objects by changing lines 103 in `yolo.py`. For example, to detect people and cars, change
```
if predicted_class != 'person':
    continue
```
to
```
if predicted_class not in ('person', 'car'):
    continue
```

## Performance
Real-time FPS with video writing:
* ~4.3fps with YOLO v3
* ~10.6fps with YOLO v4

Turning off tracking gave ~12.5fps with YOLO v4.

YOLO v4 performs much faster and appears to be more stable than YOLO v3. All tests were done using an Nvidia GTX 1070 8gb GPU
 and an i7-8700k CPU.

# Quick start
[Download](https://drive.google.com/open?id=1cewMfusmPjYWbrnuJRuKhPMwRe_b9PaT) and convert the Darknet YOLO v4 model  to a Keras model by modifying `convert.py` accordingly and run:
```
python convert.py
```
Then run demo.py:
```
python demo.py
```

## Settings

### Normal Deep SORT
By default, tracking and video writing is on and asynchronous processing is off. These can be edited in `demo.py` by changing:
```
tracking = True
writeVideo_flag = True
asyncVideo_flag = False
```

To change target file in `demo.py`:
```
file_path = 'video.webm'
```

To change output settings in `demo.py`:
```
out = cv2.VideoWriter('output_yolov4.avi', fourcc, 30, (w, h))
```

### Deep SORT with low confidence track filtering
This version has the option to hide object detections instead of tracking. The settings in `demo.py` are
```
show_detections = True
writeVideo_flag = True
asyncVideo_flag = False
```

Setting `show_detections = False` will hide object detections and show the average detection confidence and the most commonly detected class for each track.

To modify the average detection threshold, go to `deep_sort/tracker.py` and change the `adc_threshold` argument on line 40. You can also change the number of steps that the detection confidence will be averaged over by changing `n_init` here.

# Training your own models
## YOLO v4
See https://github.com/Ma-Dan/keras-yolo4.

## Deep SORT
Please note that the tracking model used here is only trained on tracking people, so you'd need to train a model yourself for tracking other objects.

See https://github.com/nwojke/cosine_metric_learning.

# Dependencies
* Tensorflow GPU 1.14
* Keras 2.3.1
* opencv-python 4.2.0
* imutils 0.5.3
* numpy 1.18.2
* sklearn

## Running with Tensorflow 1.14 vs 2.0
Navigate to the appropriate folder and run python scripts. 

### Conda environment used for Tensorflow 2.0
(see requirements.txt)
* imutils                   0.5.3                    
* keras                     2.3.1                    
* matplotlib                3.2.1                    
* numpy                     1.18.4                   
* opencv-python             4.2.0.34                 
* pillow                    7.1.2                    
* python                    3.6.10               
* scikit-learn              0.23.1                   
* scipy                     1.4.1                    
* sklearn                   0.19                     
* tensorboard               2.2.1                    
* tensorflow                2.0.0                    
* tensorflow-estimator      2.1.0                    
* tensorflow-gpu            2.2.0                    
