# AveFace
A Python-based open-source tool to generate an average face from a set of 2D facial images, using face detection, alignment, and pixel-wise averaging.
AveFace: Average Face Generator
A Python-based open-source tool to generate an average face from a set of 2D facial images.
Overview
AveFace is a simple, reproducible pipeline for creating composite average faces using open-source computer vision libraries. It takes a collection of face images, aligns them using facial landmarks, and computes a pixel-wise mean to produce a smooth average face.
Features
Batch processing of multiple 2D face images
Automatic face detection and landmark localization
Face alignment via affine transformation
Pixel-wise averaging to generate composite average face
Visualization of intermediate steps and final results
Lightweight dependencies, easy to modify and extend
Requirements
Python 3.8+
opencv-python
dlib
numpy
matplotlib
(Optional) Pillow
Install dependencies with:
bash
pip install opencv-python dlib numpy matplotlib pillow
Usage
Prepare your dataset
Place all your face images (.jpg, .png) into a single folder (e.g., ./input_faces/).
Run the pipeline
bash
python aveface.py --input ./input_faces --output ./average_face.png
View the result
The final average face will be saved to the specified output path.
How It Works
Face Detection: Detect faces in each image using a pre-trained detector.
Landmark Localization: Identify key facial landmarks (e.g., eyes, nose, mouth).
Face Alignment: Warp each face to a standard coordinate system using affine transformation based on landmarks.
Averaging: Compute the mean value for each pixel across all aligned faces.
Output: Save and visualize the resulting average face.
Example Output
(You can add an image of your generated average face here once you run the code)
License
This project is licensed under the MIT License.