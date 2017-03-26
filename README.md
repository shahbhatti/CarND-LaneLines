
# **Finding Lane Lines on the Road**
***
In this project, you will use the tools you learned about in the lesson to identify lane lines on the road.  You can develop your pipeline on a series of individual images, and later apply the result to a video stream (really just a series of images). Check out the video clip "raw-lines-example.mp4" (also contained in this repository) to see what the output should look like after using the helper functions below.

Once you have a result that looks roughly like "raw-lines-example.mp4", you'll need to get creative and try to average and/or extrapolate the line segments you've detected to map out the full extent of the lane lines.  You can see an example of the result you're going for in the video "P1_example.mp4".  Ultimately, you would like to draw just one line for the left side of the lane, and one for the right.

---
Let's have a look at our first image called 'test_images/solidWhiteRight.jpg'.  Run the 2 cells below (hit Shift-Enter or the "play" button above) to display the image.

**Note** If, at any point, you encounter frozen display windows or other confounding issues, you can always start again with a clean slate by going to the "Kernel" menu above and selecting "Restart & Clear Output".

---

**The tools you have are color selection, region of interest selection, grayscaling, Gaussian smoothing, Canny Edge Detection and Hough Tranform line detection.  You  are also free to explore and try other techniques that were not presented in the lesson.  Your goal is piece together a pipeline to detect the line segments in the image, then average/extrapolate them and draw them onto the image for display (as below).  Once you have a working pipeline, try it out on the video stream below.**

---

<figure>
 <img src="line-segments-example.jpg" width="380" alt="Combined Image" />
 <figcaption>
 <p></p>
 <p style="text-align: center;"> Your output should look something like this (above) after detecting line segments using the helper functions below </p>
 </figcaption>
</figure>
 <p></p>
<figure>
 <img src="laneLines_thirdPass.jpg" width="380" alt="Combined Image" />
 <figcaption>
 <p></p>
 <p style="text-align: center;"> Your goal is to connect/average/extrapolate line segments to get output like this</p>
 </figcaption>
</figure>


```python
#importing some useful packages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
%matplotlib inline
```

**Some OpenCV functions (beyond those introduced in the lesson) that might be useful for this project are:**

`cv2.inRange()` for color selection  
`cv2.fillPoly()` for regions selection  
`cv2.line()` to draw lines on an image given endpoints  
`cv2.addWeighted()` to coadd / overlay two images
`cv2.cvtColor()` to grayscale or change color
`cv2.imwrite()` to output images to file  
`cv2.bitwise_and()` to apply a mask to an image

**Check out the OpenCV documentation to learn about these and discover even more awesome functionality!**

Below are some helper functions to help get you started. They should look familiar from the lesson!


```python
import math

def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Or use BGR2GRAY if you read an image with cv2.imread()
    # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def region_of_interest(img, vertices):
    """
    Applies an image mask.

    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)   

    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def draw_lines(img, lines, color=[255, 0, 0], thickness=5):
    """
    NOTE: this is the function you might want to use as a starting point once you want to
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).  

    Think about things like separating line segments by their
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of
    the lines and extrapolate to the top and bottom of the lane.

    This function draws `lines` with `color` and `thickness`.    
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below

    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)  
    """    

    leftlinepts = []
    rightlinepts = []
    cleanleftlinepts = []
    cleanrightlinepts = []
    leftSlopes = []
    rightSlopes = []

    RX = []
    RY = []
    LX = []
    LY = []

    for line in lines:
        for x1,y1,x2,y2 in line:      
                if ((((y2-y1)/(x2-x1))) < 0):
                    leftSlopes.append(round(((y2-y1)/(x2-x1))*10.0))
                    leftlinepts.append((x1,y1))
                    leftlinepts.append((x2,y2))
                else:
                    rightSlopes.append(round(((y2-y1)/(x2-x1))*10.0))
                    rightlinepts.append((x1,y1))
                    rightlinepts.append((x2,y2))

    leftMostFreqS = max(set(leftSlopes), key=leftSlopes.count)
    print (leftMostFreqS)
    rightMostFreqS = max(set(rightSlopes), key=rightSlopes.count)
    print (rightMostFreqS)

    for line in lines:
        for x1,y1,x2,y2 in line:      
                if ((leftMostFreqS-1) <= (round(((y2-y1)/(x2-x1))*10.0)) <= (leftMostFreqS+1)):
                    cleanleftlinepts.append((x1,y1))
                    cleanleftlinepts.append((x2,y2))

    for line in lines:
        for x1,y1,x2,y2 in line:      
                if ((rightMostFreqS-1) <= (round(((y2-y1)/(x2-x1))*10.0)) <= (rightMostFreqS+1)):
                    cleanrightlinepts.append((x1,y1))
                    cleanrightlinepts.append((x2,y2))

    for pts in cleanleftlinepts:
        LX.append(pts[0])
        LY.append(pts[1])

    for pts in cleanrightlinepts:
        RX.append(pts[0])
        RY.append(pts[1])


    leftx1 = min(LX)
    leftx2 = max(LX)
    lefty1 = min(LY)
    lefty2 = max(LY)

    vx, vy, cx, cy = cv2.fitLine(np.array(cleanleftlinepts), cv2.DIST_L12, 0, 0.01, 0.01)

    w = abs((cy-lefty1)/vy)
    z = abs((540-cy)/vy)

    cv2.line(img, (int(cx-vx*z), int(cy-vy*z)), (int(cx+vx*w), int(cy+vy*w)), (255, 0, 0), thickness)

    rightx1 = min(RX)
    rightx2 = max(RX)
    righty1 = min(RY)
    righty2 = max(RY)

    vx, vy, cx, cy = cv2.fitLine(np.array(cleanrightlinepts), cv2.DIST_L12, 0, 0.01, 0.01)

    w = (540-cy)/vy
    z = (cy-righty1)/vy

    cv2.line(img, (int(cx-vx*z), int(cy-vy*z)), (int(cx+vx*(w)), int(cy+vy*(w))), (0, 0, 255), thickness)


def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.

    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img

# Python 3 has support for cool math symbols.

def weighted_img(img, initial_img, α=0.8, β=1., λ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.

    `initial_img` should be the image before any processing.

    The result image is computed as follows:

    initial_img * α + img * β + λ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, λ)
```

## Test on Images

Now you should build your pipeline to work on the images in the directory "test_images"  
**You should make sure your pipeline works well on these images before you try the videos.**


```python
import os
os.listdir("test_images/")
```




    ['solidYellowCurve.jpg',
     'whiteCarLaneSwitch_final.jpg',
     'solidYellowCurve_final.jpg',
     'solidWhiteRight.jpg',
     'solidWhiteRight_final.jpg',
     'whiteCarLaneSwitch.jpg',
     'solidYellowCurve2.jpg',
     'solidYellowLeft.jpg',
     'solidYellowCurve2_final.jpg',
     'solidYellowLeft_final.jpg']




```python
#reading in an image
image = mpimg.imread('test_images/whiteCarLaneSwitch.jpg')
#printing out some stats and plotting
print('This image is:', type(image), 'with dimesions:', image.shape)
plt.imshow(image)  #call as plt.imshow(gray, cmap='gray') to show a grayscaled image
```

    This image is: <class 'numpy.ndarray'> with dimesions: (540, 960, 3)





    <matplotlib.image.AxesImage at 0x7ff3f90c89b0>




![png](output_8_2.png)


run your solution on all test_images and make copies into the test_images directory).


```python
# TODO: Build your pipeline that will draw lane lines on the test_images
# then save them to the test_images directory.

#show a grayscaled image
gray = grayscale(image)
plt.imshow(gray, cmap='gray')


```




    <matplotlib.image.AxesImage at 0x7ff3f805e358>




![png](output_10_1.png)



```python
# TODO: Build your pipeline that will draw lane lines on the test_images
# then save them to the test_images directory.

#show Gaussian smoothing
blur_gray = gaussian_blur(gray, 3)

#show Canny
edges = canny(blur_gray, 50, 150)
plt.imshow(edges, cmap='gray')

```




    <matplotlib.image.AxesImage at 0x7ff3f1f8b9b0>




![png](output_11_1.png)



```python
# TODO: Build your pipeline that will draw lane lines on the test_images
# then save them to the test_images directory.

#show masked edges
masked_edges = region_of_interest(edges, np.array([[(150,540), (400,330), (550,330), (900, 540)]]))
plt.imshow(masked_edges, cmap='gray')
```




    <matplotlib.image.AxesImage at 0x7ff3f1f741d0>




![png](output_12_1.png)



```python
# TODO: Build your pipeline that will draw lane lines on the test_images
# then save them to the test_images directory.

#show hugh lines
line_img =  hough_lines(masked_edges, 2, np.pi/180, 15, 20, 20)
plt.imshow(line_img)
```

    -8.0
    6.0





    <matplotlib.image.AxesImage at 0x7ff3f1ed82b0>




![png](output_13_2.png)



```python
# TODO: Build your pipeline that will draw lane lines on the test_images
# then save them to the test_images directory.

#show lines
line_edges = weighted_img(line_img, image, 0.8, 1, 0)
plt.imshow(line_edges)
mpimg.imsave('test_images/whiteCarLaneSwitch_final.jpg', line_edges)
```


![png](output_14_0.png)


## Test on Videos

You know what's cooler than drawing lanes over images? Drawing lanes over video!

We can test our solution on two provided videos:

`solidWhiteRight.mp4`

`solidYellowLeft.mp4`


```python
# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip
from IPython.display import HTML
```


```python
#importing some useful packages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import math
from collections import Counter
%matplotlib inline

def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Or use BGR2GRAY if you read an image with cv2.imread()
    # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def region_of_interest(img, vertices):
    """
    Applies an image mask.

    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)   

    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def draw_lines(img, lines, color=[255, 0, 0], thickness=5):
    """
    NOTE: this is the function you might want to use as a starting point once you want to
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).  

    Think about things like separating line segments by their
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of
    the lines and extrapolate to the top and bottom of the lane.

    This function draws `lines` with `color` and `thickness`.    
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below

    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)  
    """    

    leftlinepts = []
    rightlinepts = []
    cleanleftlinepts = []
    cleanrightlinepts = []
    leftSlopes = []
    rightSlopes = []

    RX = []
    RY = []
    LX = []
    LY = []

    for line in lines:
        for x1,y1,x2,y2 in line:      
                if ((((y2-y1)/(x2-x1))) < 0):
                    leftSlopes.append(round(((y2-y1)/(x2-x1))*10.0))
                    leftlinepts.append((x1,y1))
                    leftlinepts.append((x2,y2))
                else:
                    rightSlopes.append(round(((y2-y1)/(x2-x1))*10.0))
                    rightlinepts.append((x1,y1))
                    rightlinepts.append((x2,y2))

    leftMostFreqS = max(set(leftSlopes), key=leftSlopes.count)
    #print (leftMostFreqS)
    rightMostFreqS = max(set(rightSlopes), key=rightSlopes.count)
    #print (rightMostFreqS)

    for line in lines:
        for x1,y1,x2,y2 in line:      
                if ((leftMostFreqS-1) <= (round(((y2-y1)/(x2-x1))*10.0)) <= (leftMostFreqS+1)):
                    cleanleftlinepts.append((x1,y1))
                    cleanleftlinepts.append((x2,y2))

    for line in lines:
        for x1,y1,x2,y2 in line:      
                if ((rightMostFreqS-1) <= (round(((y2-y1)/(x2-x1))*10.0)) <= (rightMostFreqS+1)):
                    cleanrightlinepts.append((x1,y1))
                    cleanrightlinepts.append((x2,y2))

    for pts in cleanleftlinepts:
        LX.append(pts[0])
        LY.append(pts[1])

    for pts in cleanrightlinepts:
        RX.append(pts[0])
        RY.append(pts[1])


    leftx1 = min(LX)
    leftx2 = max(LX)
    lefty1 = min(LY)
    lefty2 = max(LY)

    vx, vy, cx, cy = cv2.fitLine(np.array(cleanleftlinepts), cv2.DIST_L12, 0, 0.01, 0.01)

    w = abs((cy-lefty1)/vy)
    z = abs((540-cy)/vy)

    cv2.line(img, (int(cx-vx*z), int(cy-vy*z)), (int(cx+vx*w), int(cy+vy*w)), (255, 0, 0), thickness)

    rightx1 = min(RX)
    rightx2 = max(RX)
    righty1 = min(RY)
    righty2 = max(RY)

    vx, vy, cx, cy = cv2.fitLine(np.array(cleanrightlinepts), cv2.DIST_L12, 0, 0.01, 0.01)

    w = (540-cy)/vy
    z = (cy-righty1)/vy

    cv2.line(img, (int(cx-vx*z), int(cy-vy*z)), (int(cx+vx*(w)), int(cy+vy*(w))), (0, 0, 255), thickness)


def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.

    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img

# Python 3 has support for cool math symbols.

def weighted_img(img, initial_img, α=0.8, β=1., λ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.

    `initial_img` should be the image before any processing.

    The result image is computed as follows:

    initial_img * α + img * β + λ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, λ)

def process_image(image):

    #show a grayscaled image
    gray = grayscale(image)

    #show Gaussian smoothing
    blur_gray = gaussian_blur(gray, 3)

    #show Canny
    edges = canny(blur_gray, 50, 150)

    #show masked edges
    masked_edges = region_of_interest(edges, np.array([[(150,540), (400,330), (550,330), (900, 540)]]))

    #show hugh lines
    line_img =  hough_lines(masked_edges, 2, np.pi/180, 15, 20, 20)

    #show lines
    line_edges = weighted_img(line_img, image, 0.8, 1, 0)

    return line_edges

```

Let's try the one with the solid white lane on the right first ...


```python
white_output = 'white.mp4'
clip1 = VideoFileClip("solidWhiteRight.mp4")
white_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
%time white_clip.write_videofile(white_output, audio=False)
```

    [MoviePy] >>>> Building video white.mp4
    [MoviePy] Writing video white.mp4


    100%|█████████▉| 221/222 [00:05<00:00, 36.80it/s]


    [MoviePy] Done.
    [MoviePy] >>>> Video ready: white.mp4

    CPU times: user 5.08 s, sys: 160 ms, total: 5.24 s
    Wall time: 6.33 s


Play the video inline, or if you prefer find the video in your filesystem (should be in the same directory) and play it in your video player of choice. You may view the video at https://youtu.be/QhhzWGl7A5g


```python
HTML("""
<video width="960" height="540" controls>
  <source src="{0}">
</video>
""".format(white_output))
```





<video width="960" height="540" controls>
  <source src="white.mp4">
</video>




**At this point, if you were successful you probably have the Hough line segments drawn onto the road, but what about identifying the full extent of the lane and marking it clearly as in the example video (P1_example.mp4)?  Think about defining a line to run the full length of the visible lane based on the line segments you identified with the Hough Transform.  Modify your draw_lines function accordingly and try re-running your pipeline.**

Now for the one with the solid yellow lane on the left. This one's more tricky!


```python
yellow_output = 'yellow.mp4'
clip2 = VideoFileClip('solidYellowLeft.mp4')
yellow_clip = clip2.fl_image(process_image)
%time yellow_clip.write_videofile(yellow_output, audio=False)
```

    [MoviePy] >>>> Building video yellow.mp4
    [MoviePy] Writing video yellow.mp4


    100%|█████████▉| 681/682 [00:18<00:00, 36.52it/s]


    [MoviePy] Done.
    [MoviePy] >>>> Video ready: yellow.mp4

    CPU times: user 16.3 s, sys: 416 ms, total: 16.7 s
    Wall time: 19.4 s

You may view the video at https://youtu.be/g3h8K1j1294 

```python
HTML("""
<video width="960" height="540" controls>
  <source src="{0}">
</video>
""".format(yellow_output))
```





<video width="960" height="540" controls>
  <source src="yellow.mp4">
</video>




## Reflections

Congratulations on finding the lane lines!  As the final step in this project, we would like you to share your thoughts on your lane finding pipeline... specifically, how could you imagine making your algorithm better / more robust?  Where will your current algorithm be likely to fail?

Please add your thoughts below,  and if you're up for making your pipeline more robust, be sure to scroll down and check out the optional challenge video below!


This project was interesting and challenging for me. I learned a lot and more to learn. I need to work on "smoothing". My videos are a bit jumpy. I also need to brush up on my graphics techniques and understand python better. My current algorithm will fail on curves. My current algorithm failed miserably on the challenge video. I will continue working on it!

## Submission

If you're satisfied with your video outputs it's time to submit!  Submit this ipython notebook for review.


## Optional Challenge

Try your lane finding pipeline on the video below.  Does it still work?  Can you figure out a way to make it more robust?  If you're up for the challenge, modify your pipeline so it works with this video and submit it along with the rest of your project!


```python
challenge_output = 'extra.mp4'
clip2 = VideoFileClip('challenge.mp4')
challenge_clip = clip2.fl_image(process_image)
%time challenge_clip.write_videofile(challenge_output, audio=False)
```


```python
HTML("""
<video width="960" height="540" controls>
  <source src="{0}">
</video>
""".format(challenge_output))
```


```python

```
