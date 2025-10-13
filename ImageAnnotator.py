import supervision as sv
from ultralytics import YOLO
from PIL import Image
import gradio as gr
import numpy as np
import cv2
import urllib

# load pre-trained vision model
model = YOLO("yolo12s.pt")

def image_annotate(image:str, annotator:str) -> Image.Image:
  """
  Args:
    image: the path to the image file
    annotator: the type of annotator to use
  Returns:
    annotated image
  """
  # load the input image
  image = cv2.imread(image)

  # run object detection on the image
  result = model(image)[0]

  # convert YOLO output to a Supervision-compatible detections format
  detections = sv.Detections.from_ultralytics(result)

  # select annotator
  if annotator == "Box":
    box_annotator = sv.BoxAnnotator(thickness=6)
    annotated_image_show = box_annotator.annotate(
    scene=image.copy(),
    detections=detections)

  elif annotator == "Roundbox":
    round_box_annotator = sv.RoundBoxAnnotator(thickness=6)
    annotated_image_show = round_box_annotator.annotate(
    scene=image.copy(),
    detections=detections)

  elif annotator == "Boxcorner":
    corner_annotator = sv.BoxCornerAnnotator(thickness=6)
    annotated_image_show = corner_annotator.annotate(
    scene=image.copy(),
    detections=detections)

  elif annotator == "Color":
    color_annotator = sv.ColorAnnotator(opacity=0.7)
    annotated_image_show = color_annotator.annotate(
    scene=image.copy(),
    detections=detections)

  elif annotator == "Circle":
    circle_annotator = sv.CircleAnnotator(thickness=6)
    annotated_image_show = circle_annotator.annotate(
    scene=image.copy(),
    detections=detections)

  elif annotator == "Dot":
    dot_annotator = sv.DotAnnotator(radius=20)
    annotated_image_show = dot_annotator.annotate(
    scene=image.copy(),
    detections=detections)

  elif annotator == "Triangle":
    triangle_annotator = sv.TriangleAnnotator(height=50,base=20)
    annotated_image_show = triangle_annotator.annotate(
    scene=image.copy(),
    detections=detections)

  elif annotator == "Ellipse":
    ellipse_annotator = sv.EllipseAnnotator(thickness=6)
    annotated_image_show = ellipse_annotator.annotate(
    scene=image.copy(),
    detections=detections)

  elif annotator == "Percentage":
    percentage_bar_annotator = sv.PercentageBarAnnotator(width=80,height=20)
    annotated_image_show = percentage_bar_annotator.annotate(
    scene=image.copy(),
    detections=detections)

  elif annotator == "Heatmap":
    heatmap_annotator = sv.HeatMapAnnotator(radius=90,opacity=0.7)
    annotated_image_show = heatmap_annotator.annotate(
    scene=image.copy(),
    detections=detections)

  elif annotator == "Label":
    labels = [
    f"{class_id} {confidence:.2f}"
    for class_id, confidence
    in zip(detections.class_id, detections.confidence)
    ]

    rich_label_annotator = sv.RichLabelAnnotator(
        text_color=sv.Color.BLACK,
        text_padding=10,
        text_position=sv.Position.CENTER)

    annotated_image_show = rich_label_annotator.annotate(
        scene=image.copy(),
        detections=detections,
        labels=labels )

  elif annotator == "Blur":
    blur_annotator = sv.BlurAnnotator()
    annotated_image_show = blur_annotator.annotate(
    scene=image.copy(),
    detections=detections)

  elif annotator == "Pixelate":
    pixelate_annotator = sv.PixelateAnnotator()
    annotated_image_show = pixelate_annotator.annotate(
    scene=image.copy(),
    detections=detections)

  elif annotator == "Backgroundcolor":
    background_overlay_annotator = sv.BackgroundOverlayAnnotator()
    annotated_image_show = background_overlay_annotator.annotate(
    scene=image.copy(),
    detections=detections)

  # return annotated image
  return annotated_image_show
