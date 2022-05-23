# Robotics II - the Jetbot track follower

<p align="center">
  <img src="README_files/logo_irim.png">
</p>

## Introduction

<p align="center">
  <img src="README_files/jetbot.jpg">
</p>

## Approach

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


* `config.yml`
* `PUTDriver.py`
* `user_driving.py`
* `bot_driving.py`

## Task

### 

* model in the ONNX format
* input shape (1, 3, 224, 224)
* output shape (1, 2)

## Evaluation

<iframe width="1280" height="720" src="https://www.youtube.com/embed/XWJWwW9eQtI" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>