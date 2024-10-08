# vSlam

This vSLAM (Visual Simultaneous Localization and Mapping) implementation uses computer vision techniques to estimate the 3D structure of a scene from video input. 

### Pipeline

![Workflow](/Workflow.png)

The process begins by selecting one of three feature detectors: ORB, BRISK, or SIFT. These detectors identify key points and descriptors within each frame of the video. The user can choose which feature detection method best suits their application, balancing speed and accuracy.

Once features are detected, they are matched across frames using either the BFMatcher or FLANN-based matcher, depending on the selected detector. The matched features are refined using RANSAC to eliminate outliers, and then the relative pose (rotation and translation) between frames is estimated. 3D points are then triangulated from the inlier matches, creating a sparse 3D map of the scene. Over time, bundle adjustment is applied to optimize both the 3D points and the camera poses, improving accuracy.

This implementation outputs a 3D map of the environment and visualizes it using the Display class, with incremental updates as more frames are processed. Below is the visual representation of the workflow:

This diagram illustrates the vSLAM pipeline, from input video to feature detection, matching, pose estimation, 3D map construction, and optimization using bundle adjustment.

### Todo

Further Optimize using CUDA or Apple-metal
Optimize display. Maybe use Pangolian

