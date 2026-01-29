import torch
from torch.utils.data import DataLoader
from torchvision.models.detection import maskrcnn_resnet50_fpn
import torchvision.transforms as T
from pycocotools.coco import COCO
import os
from tqdm import tqdm

# ==============================
# Dataset loader (COCO-style)
# ==============================
class NucleusDataset(torch.utils.data.Dataset):
    def __init__(self, root, annFile, transforms=None):
        self.root = root
        self.coco = COCO(annFile)
        self.ids = list(self.coco.imgs.keys())
        self.transforms = transforms

    def __getitem__(self, index):
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)
        path = coco.loadImgs(img_id)[0]['file_name']

        # load image
        import cv2
        import numpy as np
        img = cv2.imread(os.path.join(self.root, path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # load masks
        masks = []
        boxes = []
        for ann in anns:
            rle = coco.annToRLE(ann)
            mask = coco.annToMask(ann)
            masks.append(mask)
            bbox = ann['bbox']
            x, y, w, h = bbox
            boxes.append([x, y, x+w, y+h])

        masks = torch.as_tensor(masks, dtype=torch.uint8) if masks else torch.zeros((0, img.shape[0], img.shape[1]), dtype=torch.uint8)
        boxes = torch.as_tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4), dtype=torch.float32)

        labels = torch.ones((len(anns),), dtype=torch.int64)  # all nuclei are "1"

        target = {"boxes": boxes, "labels": labels, "masks": masks, "image_id": torch.tensor([img_id])}

        if self.transforms:
            img = self.transforms(img)

        return img, target

    def __len__(self):
        return len(self.ids)


# ==============================
# Training setup
# ==============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paths
images_dir = "data/images"
annotations_file = "data/annotations.json"

# Dataset & DataLoader
dataset = NucleusDataset(images_dir, annotations_file, transforms=T.ToTensor())
data_loader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=lambda x: tuple(zip(*x)), num_workers=0)

# Model
model = maskrcnn_resnet50_fpn(weights="DEFAULT")
num_classes = 2  # background + nucleus
in_features = model.roi_heads.box_predictor.cls_score.in_features
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, 256, num_classes)

model.to(device)

# Optimizer
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

# ==============================
# Training loop with checkpoint
# ==============================
num_epochs = 5
checkpoint_path = "checkpoint.pth"

# Resume if checkpoint exists
start_epoch = 0
if os.path.exists(checkpoint_path):
    print("Loading checkpoint...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    start_epoch = checkpoint["epoch"] + 1
    print(f"Resuming from epoch {start_epoch}")

for epoch in range(start_epoch, num_epochs):
    model.train()
    epoch_loss = 0
    for images, targets in tqdm(data_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
        images = list(img.to(device) for img in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        epoch_loss += losses.item()

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

    print(f"Epoch {epoch+1} finished. Loss: {epoch_loss:.4f}")

    # Save checkpoint
    torch.save({
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict()
    }, checkpoint_path)
    print(f"Checkpoint saved at epoch {epoch+1}")