# Introduction
This project was inspired by:
* https://github.com/nwojke/deep_sort
* https://github.com/Ma-Dan/keras-yolo4
* https://github.com/Qidian213/deep_sort_yolov3

I swapped out YOLO v3 for YOLO v4 and added the option for asynchronous processing, which significantly improves
the FPS. However, FPS monitoring is disabled when asynchronous processing is used since it isn't accurate.

## YOLO v3 and YOLO v4 comparison video with Deep SORT
[![Comparison Video Link](https://img.youtube.com/vi/_8WkO3hVOlY/0.jpg)](https://youtu.be/_8WkO3hVOlY)

The white boxes are Deep SORT trackers and the blue boxes are YOLO v4 detections. Each white box has a tracking ID at the top and each blue box has a YOLO detection confidence score at the bottom. 

## With asynchronous processing
![](async_example.gif)

As you can see in the gif, asynchronous processing has better FPS but causes stuttering.

This code only detects and tracks people, but can be changed to detect other objects by changing lines 103 in yolo.py. For example, to detect people and cars, change
```
if predicted_class != 'person':
    continue
```
to
```
if predicted_class not in ('person', 'car'):
    continue
```

Please note that Deep SORT is only trained on tracking people, so you'd need to train a model yourself for tracking other objects.
See https://github.com/nwojke/cosine_metric_learning.

## Performance
Real-time FPS with video writing:
* ~4.3fps with YOLO v3
* ~10.6fps with YOLO v4

Turning off tracking gave ~12.5fps with YOLO v4.

YOLO v4 performs much faster and appears to be more stable than YOLO v3. All tests were done using an Nvidia GTX 1070 8gb GPU
 and an i7-8700k CPU.

# Quick start
[Download](https://drive.google.com/open?id=1cewMfusmPjYWbrnuJRuKhPMwRe_b9PaT) and convert the Darknet YOLO v4 model  to a Keras model by modifying convert.py accordingly and run:
```
python convert.py
```
Then run demo.py:
```
python demo.py
```

## Settings
By default, tracking and video writing is on and asynchronous processing is off. These can be edited in demo.py by changing:
```
tracking = True
writeVideo_flag = True
asyncVideo_flag = False
```

To change target file in demo.py:
```
file_path = 'video.webm'
```

To change output settings in demo.py:
```
out = cv2.VideoWriter('output_yolov4.avi', fourcc, 30, (w, h))
```

# Training your own YOLO v4 model
See https://github.com/Ma-Dan/keras-yolo4.

# Dependencies
* Tensorflow GPU 1.14
* Keras 2.3.1
* opencv-python 4.2.0
* imutils 0.5.3
* numpy 1.18.2
* sklearn

# Upgrading to run with Tensorflow 2.0 YOLO v4 model

Changes in files:

Compatibility with Tf1 – tools/generate_detections.py

    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()

Deprecated linear assignment on new versions - deep_sort/linear_assignment.py

    from scipy.optimize import linear_sum_assignment

    indices = linear_sum_assignment(cost_matrix)
    indices = np.asarray(indices)
    indices = np.transpose(indices)

Conda environmnet with the following versions:

# Name                    Version                   Build  Channel
imutils                   0.5.3                    pypi_0    pypi
keras                     2.3.1                    pypi_0    pypi
matplotlib                3.2.1                    pypi_0    pypi
numpy                     1.18.4                   pypi_0    pypi
opencv-python             4.2.0.34                 pypi_0    pypi
pillow                    7.1.2                    pypi_0    pypi
python                    3.6.10               h7579374_2
scikit-learn              0.23.1                   pypi_0    pypi
scipy                     1.4.1                    pypi_0    pypi
sklearn                   0.0                      pypi_0    pypi
tensorboard               2.2.1                    pypi_0    pypi
tensorflow                2.0.0                    pypi_0    pypi
tensorflow-estimator      2.1.0                    pypi_0    pypi
tensorflow-gpu            2.2.0                    pypi_0    pypi


