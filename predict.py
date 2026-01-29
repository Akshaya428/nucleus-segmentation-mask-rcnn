import torch
import torchvision.transforms as T
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from PIL import Image
import numpy as np
import cv2
import sys

# ==============================
# Config
# ==============================
CHECKPOINT_PATH = "checkpoint.pth"   # your trained checkpoint
IMAGE_PATH = "C:\\Users\\aksha\\Projects\\maskrcnn_nucleus\\stage1_train\\1e8408fbb1619e7a0bcdd0bcd21fae57e7cb1f297d4c79787a9d0f5695d77073\\images\\1e8408fbb1619e7a0bcdd0bcd21fae57e7cb1f297d4c79787a9d0f5695d77073.png"       # replace with your image path
OUTPUT_PATH = "prediction_result.png"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==============================
# Load model
# ==============================
print("Loading model...")
model = maskrcnn_resnet50_fpn(weights=None)

# Adjust for 2 classes (background + nucleus)
num_classes = 2
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, 256, num_classes)

# Load checkpoint
checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
model.load_state_dict(checkpoint["model"])
model.to(device).eval()
print("✅ Model loaded")

# ==============================
# Load image
# ==============================
img = Image.open(IMAGE_PATH).convert("RGB")
transform = T.ToTensor()
img_tensor = transform(img).to(device)

# Convert to OpenCV (black background for drawing)
img_cv = np.zeros((img.size[1], img.size[0], 3), dtype=np.uint8)

# ==============================
# Inference
# ==============================
with torch.no_grad():
    prediction = model([img_tensor])[0]

# ==============================
# Draw predictions (red boxes + contours)
# ==============================
for box, mask, score in zip(prediction["boxes"], prediction["masks"], prediction["scores"]):
    if score < 0.5:  # filter weak predictions
        continue

    # Bounding box
    x1, y1, x2, y2 = box.int().tolist()
    cv2.rectangle(img_cv, (x1, y1), (x2, y2), (0, 0, 255), 2)

    # Mask -> contour
    mask = mask[0].cpu().numpy()
    mask = (mask > 0.5).astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img_cv, contours, -1, (0, 0, 255), 2)

# ==============================
# Save and Exit
# ==============================
cv2.imwrite(OUTPUT_PATH, img_cv)
print(f"✅ Prediction saved at {OUTPUT_PATH}")

sys.exit(0)
