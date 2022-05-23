# Robotics II - the Jetbot track follower

<p align="center">
  <img src="README_files/logo_irim.png">
</p>

## Introduction

<p align="center">
  <img src="README_files/jetbot.jpg">
</p>

## Approach

### The short story of NVIDIA end2end driving (25 Apr 2016)

["End to End Learning for Self-Driving Cars"](https://arxiv.org/pdf/1604.07316.pdf)

<p align="center">
  <img src="README_files/nvidia_01.png">
</p>

<p align="center">
  <img src="README_files/nvidia_02.png">
</p>

<p align="center">
  <img src="README_files/nvidia_03.png">
</p>

### Our clone of the approach

<p align="center">
  <img src="README_files/signals.jpg">
</p>

## Dataset


|    **Kind**   	| **Input image** 	| **Forward signal** 	| **Left signal** 	|
|:-------------:	|:---------------:	|:------------------:	|:---------------:	|
| Forward drive 	| ![forward image](README_files/0158.jpg) | 0.9921875	| 0.0 |
|   Left curve  	| ![left curve](README_files/0180.jpg) | 0.453125	| 0.6328125	|
|  Right curve  	| ![right curve](README_files/0219.jpg)  | 0.85	| -1.0	|

## Scripts


* `config.yml`:
  ```yaml
  model:
    path: ''
  ```
  ```yaml
  robot:
    max_speed: 0.22
    max_steering: 0.5
  ```
  ```yaml
    differential:
        left: 0.0
        right: 0.0
  ```

    | **Vehicle** 	| **robot.differential.left** 	| **robot.differential.right** 	|
    |:-----------:	|:---------------------------:	|:----------------------------:	|
    |  Jetbot 01  	|             1.0             	|              0.9             	|
    |  Jetbot 02  	|                             	|                              	|
    |  Jetbot 03  	|                             	|                              	|

* `PUTDriver.py`
* `user_driving.py`
* `bot_driving.py`

## Task

### 

* model in the ONNX format
* input shape (1, 3, 224, 224)
* output shape (1, 2)
* preprocess and postprocesses functions

## Evaluation

[![example](https://img.youtube.com/vi/oGQLA6oU2p4/0.jpg)](https://www.youtube.com/watch?v=oGQLA6oU2p4)