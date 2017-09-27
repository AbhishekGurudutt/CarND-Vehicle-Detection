## Vehicle Detection Project

[//]: # (Image References)
[image1]: ./output_images/dataset.png
[image2]: ./output_images/hog_feature.png
[image3]: ./output_images/sliding_window.png
[image4]: ./output_images/car_detection.png
[image5]: ./output_images/video_pipeline.png
[image6]: ./output_images/labelled_pipeline.png
[image7]: ./output_images/final_image.png


---

`VehicleDetection.ipynb` file contains the code to this project.

### Histogram of Oriented Gradients (HOG)
#### 1. Reading data
Cell 2 contains the code to read all the image dataset. The image dataset is provided by Udacity. It contains two folders under *training* folder. *vehicles* folder contains dataset of all the vehicles. *non-vehicles* folder contain dataset of all non-vehicles.

package `glob` is used to read all the image dataset. Cell 3 is used to display few images of car and non-car images.

![alt text][image1]

#### 2. HOG Parameter tuning

The code for this step is contained in the 4th code cell. Different color spaces and different parameters were explored. The final values were considered according to best validation accuracy from the classifier used in this project.

Initially, hog_channel = 0, spatial_size = (16, 16), hist_bins = 16, orientations = 9, 8 pixels per cell and 2 cells per block were used and following is the output observed from the classifier.

![alt text][image2]

| Color Space | Training time | Accuracy |
| :----: | :----: |  :----: |
| LUV | 4.25 | 0.9854 |
| RGB | 5.22 | 0.9657 |
| YUV | 4.39 | 0.9703 |
| HSV | 6.13 | 0.9575 |
| HLS | 6.35 | 0.9589 |
| YCrCb | 4.55 | 0.9921 |

YCrCb color space provided the best validation accuracy, and was used in this project.

#### 3. Training the classifier

Code block 7 shows the code for training the classifier. Support Vector Machine, linear SVC is used for classification. This classification is fed with hog features, color histogram and spatial features. Model accuracy of 99.21% with 0.00346 seconds to predict 10 samples was obtained.

The parameters for final model was
```python
color_space = 'YCrCb'
hog_channel = "ALL"
spatial_size = (16, 16)
hist_bins = 16
orient = 9  
pix_per_cell = 8
cell_per_block = 2
```

### Sliding Window Search

#### 1. Overview

The code is implemented such that, hog features and training model parameters are computed and saved once, and then sub-sampled to get all of the overlaying windows. There are multiple windows defined, with different scaling factor. With window of scaling factor 1 would result in a window that's 8 x 8 cells then the overlap of each window is in terms of the cell distance. This means that a cells_per_step = 2 would result in a search window overlap of 75%.

Scales of 1.5, 1.25, and 1.0 were used with y-axis ranges of(500,680),(400, 600),and (380,460).

![alt text][image3]

#### 2. Examples of test images to demonstrate pipeline

For detecting cars, I searched using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image4]

---

### Video Implementation

Here's a [link to my video result](./video_output.mp4)


#### 1. Filter for false positives and method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:

![alt text][image5]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][image6]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7]


---

### Discussion

* The window size was chosen to decrease with distance from car in Y direction, but not taken car in x direction. This is one part where I will work on and improve the optimization.
* The algorithm does not detect well with bright images and this can be solved with better classifier training.
* When there is an overlap with the car, the algorithm does not detect the one behind. This can be detected by having a memory of previous detected car.
