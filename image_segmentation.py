import torch
from PIL import Image
from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation
import numpy as np
from torchvision.transforms import functional as F
import matplotlib.pyplot as plt

def load_model():
    # Load feature extractor and model from Hugging Face
    feature_extractor = SegformerFeatureExtractor.from_pretrained("nvidia/segformer-b0-finetuned-cityscapes-1024-1024")
    model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b0-finetuned-cityscapes-1024-1024")
    return feature_extractor, model

def preprocess_image(image_path, feature_extractor):
    # Load image and preprocess
    image = Image.open(image_path).convert("RGB")
    inputs = feature_extractor(images=image, return_tensors="pt")
    return inputs['pixel_values']

def segment_image(pixel_values, model):
    # Perform segmentation
    model.eval()
    with torch.no_grad():
        outputs = model(pixel_values)
    return outputs.logits

def save_segmented_image(logits, output_path):
    logits = logits[0]  # we only processed one image
    seg_image = torch.argmax(logits, dim=0)
    seg_image = seg_image.detach().cpu().numpy()

    # Create a color map
    cmap = plt.get_cmap('tab20')  # You can choose a colormap that fits your labels
    max_label = seg_image.max() + 1
    colors = cmap(list(range(max_label)))

    # Map each label to a color
    colorized_image = np.zeros((seg_image.shape[0], seg_image.shape[1], 3), dtype=np.uint8)
    for label in range(max_label):
       mask = seg_image == label
       colorized_image[mask] = np.array(colors[label][:3]) * 255

    # Convert array to PIL Image and save
    colorized_image = Image.fromarray(colorized_image, 'RGB')
    colorized_image.save(output_path)

def process_image(image_path, output_path):
    feature_extractor, model = load_model()
    pixel_values = preprocess_image(image_path, feature_extractor)
    logits = segment_image(pixel_values, model)
    save_segmented_image(logits,output_path)