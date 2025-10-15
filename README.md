# **RoboFlow-Supervision**

This project is a computer vision application that provides a user-friendly interface for detecting objects and human poses in images. Built with Gradio, YOLO, and Supervision, it offers a range of annotation styles to visualize the detection results.

![Object detection](Detection.jpg)

## **Features**

### **Object Detection**
The object detection module can identify and annotate objects in an image with the following styles:
- **Box**: A standard bounding box around the detected object.
- **Roundbox**: A bounding box with rounded corners.
- **Boxcorner**: Highlights the corners of the bounding box.
- **Color**: Applies a color mask to the detected object.
- **Circle**: Places a circle around the detected object.
- **Dot**: Marks the center of the detected object with a dot.
- **Triangle**: Places a triangle marker on the detected object.
- **Ellipse**: Draws an ellipse around the detected object.
- **Percentage**: Displays a percentage bar indicating the detection confidence.
- **Heatmap**: Generates a heatmap to visualize object locations.
- **Label**: Displays the class name and confidence score of the detected object.
- **Blur**: Blurs the area of the detected object.
- **Pixelate**: Pixelates the detected object.
- **Backgroundcolor**: Overlays a colored background on the detected object.

### **Pose Detection**
The pose detection module can identify human keypoints and annotate them in the following ways:
- **Vertex**: Marks the keypoints (e.g., shoulders, elbows, knees) with vertices.
- **Edge**: Connects the keypoints with edges to visualize the pose.
- **Vertexlabel**: Labels the keypoints with their corresponding names.

## **Technologies Used**
- **Gradio**: For creating the interactive web-based user interface.
- **YOLO**: For the underlying object and pose detection models.
- **Supervision**: For annotating the detection results.
- **Docker**: For containerizing the application for easy local deployment.

## **Getting Started**

### **Prerequisites**
- Docker must be installed on your local machine.

### **Running the Application**
1. **Clone the repository:**
   ```bash
   git clone https://github.com/Ifeanyi/RoboFlow-Supervision.git
   cd RoboFlow-Supervision
   ```
2. **Build the Docker image:**
   ```bash
   docker build -t roboflow-supervision .
   ```
3. **Run the Docker container:**
   ```bash
   docker run -p 7860:7860 roboflow-supervision
   ```
4. **Access the application:**
   Open your web browser and navigate to `http://localhost:7860`.

Please give this project a ⭐ if you like it.