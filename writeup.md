## Vehicle Detection Project

[//]: # (Image References)
[image1]: ./output_images/dataset.png
[image2]: ./output_images/HOG_feature.png
[image3]: ./examples/sliding_windows.jpg
[image4]: ./examples/sliding_window.jpg
[image5]: ./examples/bboxes_and_heat.png
[image6]: ./examples/labels_map.png
[image7]: ./examples/output_bboxes.png
[video1]: ./project_video.mp4

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

I implemented the efficient sliding window search where I can extract hog features once and then sub-sample to get all of its overlaying windows. Each window is defined by a scaling factor where a scale of 1 would result in a window that's 8 x 8 cells then the overlap of each window is in terms of the cell distance. This means that a cells_per_step = 2 would result in a search window overlap of 75%.

I tried 3 scales at different y-positions. scales = [1.5, 1.25, 1.0] y_start_stops = [(500,680),(400, 600),(380,460)], colors = [blue, green, red]

![alt text][image3]

####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image4]
---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video.mp4)


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:

![alt text][image5]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][image6]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7]



---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  
