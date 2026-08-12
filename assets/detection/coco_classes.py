"""
The standard 80 COCO object classes, in COCO's own canonical class-index
order (0-79) - the same order Ultralytics' COCO-pretrained checkpoints
report through model.names, so COCO_CLASSES[i] always lines up with
class_id i in a detection result.

Powers the "COCO Classes" picker (gui/coco_classes_window.py): pick a
stock COCO-pretrained model, let the user check whichever of these 80
classes they care about, and the app only shows detections for that set.
"""

# Ultralytics auto-downloads this from its own GitHub Releases the first
# time it's requested and caches it locally after - a small, general-
# purpose COCO detector, good default for "just show me some COCO
# classes" without the user needing to go find or train a weight file.
COCO_MODEL_NAME = "yolov8n.pt"

COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
    "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
]

assert len(COCO_CLASSES) == 80
