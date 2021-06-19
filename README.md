# MetaSCI

This code is for our CVPR 2021 paper "MetaSCI: Scalable and Adaptive Reconstruction for Video Compressive Sensing".

This is the initial version. We will continue to update it. 



### Environment

```
tensorflow 1.3
```





### Directory

- `dataset\mask` : encoding mask, shape = [M N Cr NUM], i.e., NUM  [M N Cr] masks stacked along the last dimension.





### Parameter

- `datadir`:  path for training set (orig image frames) directory
- `maskpath`: path for encoding mask  file

