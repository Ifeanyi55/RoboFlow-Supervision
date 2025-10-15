# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container at /app
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Download YOLO models
RUN curl -L "https://github.com/autodistill/autodistill-yolov8/releases/download/v0.1.1/yolov8s.pt" -o yolo12s.pt
RUN curl -L "https://github.com/autodistill/autodistill-yolov8/releases/download/v0.1.1/yolov8s-pose.pt" -o yolo11s-pose.pt

# Copy the rest of the application's code
COPY . .

# Make port 7860 available to the world outside this container
EXPOSE 7860

# Run app.py when the container launches
CMD ["python", "app.py"]