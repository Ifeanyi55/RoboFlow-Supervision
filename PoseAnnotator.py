import supervision as sv
from ultralytics import YOLO
from PIL import Image
import gradio as gr
import numpy as np
import cv2
import urllib

# load pre-trained vision model
model2 = YOLO("yolo11s-pose.pt")

# build pose annotator
def pose_annotate(image:str, pose_annotator:str) -> Image.Image:
  """
  Args:
    image: the path to the image file
    pose_annotator: the type of annotator to use
  Returns:
    annotated image
  """
  # load the input image
  image = cv2.imread(image)

  # run object detection on the image
  result = model2(image)[0]

  # detect keypoints in image
  key_points = sv.KeyPoints.from_ultralytics(result)

  if pose_annotator == "Vertex":
    vertex_annotator = sv.VertexAnnotator(
    color=sv.Color.GREEN,
    radius=10
    )

    annotated_frame = vertex_annotator.annotate(
    scene=image.copy(),
    key_points=key_points
    )

  elif pose_annotator == "Edge":
    edge_annotator = sv.EdgeAnnotator(
    color=sv.Color.GREEN,
    thickness=5
    )
    annotated_frame = edge_annotator.annotate(
        scene=image.copy(),
        key_points=key_points
    )

  elif pose_annotator == "Vertexlabel":
    vertex_label_annotator = sv.VertexLabelAnnotator(
    color=sv.Color.GREEN,
    text_color=sv.Color.BLACK,
    border_radius=5
    )
    annotated_frame = vertex_label_annotator.annotate(
        scene=image.copy(),
        key_points=key_points
    )

  return annotated_frame
