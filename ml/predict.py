import os
import argparse
import numpy as np
from PIL import Image
import torch
import torchvision.transforms.functional as TF

from model import UNet

# Land Cover Class Color Map
# Class 0: Built-up (Reddish-brown / Slate grey) -> [150, 75, 75]
# Class 1: Vegetation (Forest Green) -> [34, 139, 34]
# Class 2: Water (Deep Blue) -> [30, 100, 200]
COLOR_MAP = np.array([
    [150, 75, 75],   # Class 0: Built-up
    [34, 139, 34],   # Class 1: Vegetation
    [30, 100, 200]   # Class 2: Water
], dtype=np.uint8)

CLASS_NAMES = {
    0: "Built-up (Urban)",
    1: "Vegetation (Green Cover)",
    2: "Water"
}

def load_model(model_path, device):
    """
    Instantiates U-Net and loads model weights.
    """
    model = UNet(n_channels=3, n_classes=3)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    # Check if checkpoint is a dictionary from train.py or direct state dict
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
        
    model.to(device)
    model.eval()
    return model

def preprocess_image(image_path, size=(256, 256)):
    """
    Loads and normalizes image for UNet forward pass.
    """
    original_img = Image.open(image_path).convert('RGB')
    resized_img = original_img.resize(size)
    
    # Standardize image to PyTorch tensor format
    img_tensor = TF.to_tensor(resized_img)
    # ImageNet normalization
    img_tensor = TF.normalize(img_tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    return img_tensor.unsqueeze(0), original_img.size # Return tensor + original dimensions

def run_inference(model, img_tensor, device):
    """
    Runs model inference and returns the predicted class mask.
    """
    with torch.no_grad():
        img_tensor = img_tensor.to(device)
        logits = model(img_tensor) # Shape: [1, 3, H, W]
        probs = torch.softmax(logits, dim=1)
        pred_mask = torch.argmax(probs, dim=1).squeeze(0) # Shape: [H, W]
        
    return pred_mask.cpu().numpy()

def color_code_mask(pred_mask):
    """
    Maps 2D class-index mask to a 3D RGB color mask.
    """
    h, w = pred_mask.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    
    for c in range(3):
        color_mask[pred_mask == c] = COLOR_MAP[c]
        
    return Image.fromarray(color_mask)

def calculate_stats(pred_mask):
    """
    Calculates pixel count and percentage distribution of each class.
    """
    total_pixels = pred_mask.size
    unique, counts = np.unique(pred_mask, return_counts=True)
    counts_dict = dict(zip(unique, counts))
    
    stats = {}
    for c in range(3):
        count = counts_dict.get(c, 0)
        percentage = (count / total_pixels) * 100
        stats[c] = {
            'name': CLASS_NAMES[c],
            'count': count,
            'percentage': percentage
        }
    return stats

def format_stats_output(stats, label=""):
    out = [f"=== Land Cover Statistics {label} ==="]
    for c, data in stats.items():
        out.append(f"- {data['name']}: {data['count']} px ({data['percentage']:.2f}%)")
    return "\n".join(out)

def compare_land_cover(stats_t1, stats_t2, label_t1="T1", label_t2="T2"):
    """
    Calculates absolute and relative change in land cover classes over time.
    """
    out = [f"=== Land Cover Change Detection Analysis ({label_t1} -> {label_t2}) ==="]
    for c in range(3):
        p1 = stats_t1[c]['percentage']
        p2 = stats_t2[c]['percentage']
        diff = p2 - p1
        name = CLASS_NAMES[c]
        
        sign = "+" if diff >= 0 else ""
        out.append(f"- {name}: {p1:.2f}% -> {p2:.2f}% (Change: {sign}{diff:.2f}%)")
        
    return "\n".join(out)

def main():
    parser = argparse.ArgumentParser(description="Run Land Cover Inference and Change Detection")
    parser.add_argument('--image', type=str, required=True, help='Path to satellite image')
    parser.add_argument('--image_compare', type=str, default=None, help='Path to second satellite image for comparison')
    parser.add_argument('--model_path', type=str, default='ml/checkpoints/best_model.pth', help='Path to best U-Net checkpoint')
    parser.add_argument('--output', type=str, default='ml/output_mask.png', help='Path to save prediction mask')
    parser.add_argument('--output_compare', type=str, default='ml/output_mask_compare.png', help='Path to save secondary prediction mask')
    
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    
    if not os.path.exists(args.model_path):
        print(f"Error: Model checkpoint not found at {args.model_path}. Please run train.py first.")
        return
        
    print(f"Loading U-Net model from {args.model_path}...")
    model = load_model(args.model_path, device)
    
    # Process First Image
    print(f"Processing image: {args.image}...")
    img_tensor, orig_size = preprocess_image(args.image)
    pred_mask = run_inference(model, img_tensor, device)
    
    # Save first mask
    color_mask = color_code_mask(pred_mask)
    # Resize back to original size for exact matching if desired
    color_mask_resized = color_mask.resize(orig_size, Image.Resampling.NEAREST)
    color_mask_resized.save(args.output)
    print(f"Saved classification mask to: {args.output}")
    
    # Stats for first image
    stats_t1 = calculate_stats(pred_mask)
    print(format_stats_output(stats_t1, "(Image 1)"))
    
    # Process Second Image if provided
    if args.image_compare:
        print(f"\nProcessing second image (comparison): {args.image_compare}...")
        img_tensor_comp, orig_size_comp = preprocess_image(args.image_compare)
        pred_mask_comp = run_inference(model, img_tensor_comp, device)
        
        # Save second mask
        color_mask_comp = color_code_mask(pred_mask_comp)
        color_mask_comp_resized = color_mask_comp.resize(orig_size_comp, Image.Resampling.NEAREST)
        color_mask_comp_resized.save(args.output_compare)
        print(f"Saved comparison classification mask to: {args.output_compare}")
        
        # Stats for second image
        stats_t2 = calculate_stats(pred_mask_comp)
        print(format_stats_output(stats_t2, "(Image 2)"))
        
        # Compare land cover changes
        comparison_results = compare_land_cover(stats_t1, stats_t2, "T1 (2020)", "T2 (2026)")
        print("\n" + comparison_results)
        
        # Save change stats to text file
        stats_txt_path = os.path.splitext(args.output)[0] + "_change_report.txt"
        with open(stats_txt_path, 'w') as f:
            f.write(format_stats_output(stats_t1, "T1 (2020)") + "\n\n")
            f.write(format_stats_output(stats_t2, "T2 (2026)") + "\n\n")
            f.write(comparison_results)
        print(f"Saved analysis change report to: {stats_txt_path}")

if __name__ == '__main__':
    main()
