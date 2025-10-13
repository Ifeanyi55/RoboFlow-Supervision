from ImageAnnotator import image_annotate
from PoseAnnotator import pose_annotate
import gradio as gr

objectDetector = gr.Interface(
    fn = image_annotate,
    theme = "gstaff/sketch",
    title="Object Detection",
    inputs = [gr.Image(type="filepath",label="Image"), gr.Radio(label="Select Annotator",
                                                 choices=["Box","Roundbox","Boxcorner","Color","Circle","Dot","Triangle",
                                                          "Ellipse","Percentage","Heatmap","Label","Blur",
                                                          "Pixelate","Backgroundcolor"],
                                                 value = "Box")],
    outputs = gr.Image(label = "Annotated Image"),
    examples = [["cars.jpg","Box"],
                ["colorful-backgrounds-for-laptops.jpg","Roundbox"],
                ["final_animals-homeschooling_credit-alamy.jpg","Circle"],
                ["Furry.png","Pixelate"],
                ["Desktop-Wallpaper-HD4.jpeg","Label"],
                ["bowling_ball.jpg","Percentage"]]
)

poseDetector = gr.Interface(
    fn = pose_annotate,
    theme = "gstaff/sketch",
    title="Pose Detection",
    inputs = [gr.Image(type="filepath",label="Image"), gr.Radio(label="Select Annotator",
                                                 choices=["Vertex","Edge","Vertexlabel"],
                                                 value = "Vertex")],
    outputs = gr.Image(label = "Annotated Image"),
    examples = [["Pose.jpg", "Edge"],["black runner.jpg", "Vertexlabel"]]
)

app = gr.TabbedInterface([objectDetector,poseDetector],["Object Detection","Pose Detection"], theme = "gstaff/sketch")

if __name__ == "__main__":
  app.launch(mcp_server = True)
