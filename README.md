# NeuralNetToy01

**This is a work in progress**

http://keras.io

*Scaling Up Images with Numpy and Scipy*

*Read, scale up, and save PNG image data x4 in numpy and scipy*
http://stackoverflow.com/questions/7525214/how-to-scale-a-numpy-array

```python
import numpy as np
from scipy.misc import imread, imsave
img = imread('myimage.png')
n = 4
imgx4 = np.kron(img, np.ones((n,n)))
imsave('myimagex4.png', imgx4)
```

*Movie Making*

*Ubuntu has started using the libav fork of ffmpeg, so you can use the avconv utility:*
http://askubuntu.com/questions/432542/is-ffmpeg-missing-from-the-official-repositories-in-14-04

*Example command line for creating a video of png images:*
$ avconv -r 10 -start_number 1 -i "movie/%05d.png" out.mp4

