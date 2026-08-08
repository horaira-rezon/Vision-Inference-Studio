VISION_TASKS = [
    ("classification", "Image Classification"),
    ("detection", "Object Detection"),
    ("instance_segmentation", "Instance Segmentation"),
    ("semantic_segmentation", "Semantic Segmentation"),
    ("pose", "2D Pose Estimation"),
]

ARCHITECTURES = {
    "classification": [
        ("efficientnet", "EfficientNet"),
        ("resnet", "ResNet"),
    ],
    "detection": [
        ("yolo", "YOLO"),
        ("rf_detr", "RF-DETR"),
        ("rcnn", "R-CNN"),
        ("faster_rcnn", "Faster R-CNN"),
    ],
    "instance_segmentation": [
        ("yolo", "YOLO"),
        ("sam", "SAM"),
        ("rf_detr", "RF-DETR"),
        ("faster_rcnn", "Mask R-CNN"),
    ],
    "semantic_segmentation": [
        ("unet", "U-Net"),
    ],
    "pose": [
        ("yolo", "YOLO"),
        ("rf_detr", "RF-DETR"),
    ],
}

TRACKING_OPTIONS = [
    ("none", "No Tracking"),
    ("bytetrack", "ByteTrack"),
    ("botsort", "BotSORT"),
    ("ocsort", "OC-SORT"),
    ("deepocsort", "DeepOC-SORT"),
]

def architecture_choices(task):
    return ARCHITECTURES.get(task, [])
