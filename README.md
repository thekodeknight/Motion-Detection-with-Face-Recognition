# Motion-Detection-with-Face-Recognition
This is a project created using opencv and python which will detect motion and will recognize faces in the frame and and concurrently will be writing the frames to a video file.

## How it works:
    Motion Detection:
          We can detect motion between two consecutive frames by finding the absolute difference between two frame and we will generating a threshold image of the diffenrence image and then we will find the contours for edge detection.
          
    Face Recognition:
          When motion will be detected, recording will start and simultaneously face recogniton will start. 

## Steps for runnign the proect:

    Step 1:
        At first, we need to create our dataset and training the porgram to recognize the face by runnning the 'lbp_circular.py' file.

    Step 2:
        Then we need to run 'rec_write.py' file to run our project.
  
