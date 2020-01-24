# eFaceDetection - Embedded Face Detection 
Face detection algorithms for microchip like STM32.

## Algorithms:
* viola_jones - 2001   
Feature: Haar.  
Method: Adaboost.  
Dataset: http://cbcl.mit.edu/software-datasets/FaceData2.html  

* pico - 2014  
Feature: Binary tree with pixel intensity comparisons.   
Method: Gentle Boost.  
Ref:
Object Detection with Pixel Intensity Comparisons Organized in Decision Trees

* npd - 2014   
Feature: Normalized pixel difference(NPD) 
Method: Deep quadratic tree. 
Ref:
A Fast and Accurate Unconstrained Face Detector  


* opencv   
python3 + opencv implmentation of viola_jones algorithm.
