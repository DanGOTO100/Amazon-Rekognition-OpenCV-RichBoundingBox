# RekognitionOpenCVNumPy

Rekognition, NumPy and OpenCV integrated providing rich visual video overlays to input video

Amazon Rekognition is a deep learning-based service for image and video analysis. It allows us to extract a rich set of metadata from both images and videos easily, through its set of APIs.

Among many of Amazon Rekognition’s great features, it provides the ability  to detect people in images and videos. We can create a set of reference images of persons, a “face collection”, as a private repository of face images. With Amazon Rekognition we can detect if any those persons in the set appear in any given image or video.

What we are going to do in this post is to contextualize the detection output when searching for persons, in our face collection, within a given video.  

With this contextualization, we make the output of the detection easier to understand and consume, converting it in visual information as an overlay to the video itself. Also, we will have new visual insights about what has been detected in the video, providing not only a new way to use the detection’s output, but also opening a whole set of new uses cases and applications for Rekognition in several other industry verticals.

We are going to display a bounding box around the person’s face detected in the video, some labels like similarity (confidence that the face detected and the one in the face collection are the same), frame and coordinates, and also the reference image used in our collection to detect the matched face.  We will display all this information x each time the detected person is in the video, as an added layer to the video.
