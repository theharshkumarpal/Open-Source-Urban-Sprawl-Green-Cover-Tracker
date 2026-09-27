import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from model import UNet
from dataset import SatelliteDataset, JointTransform, generate_synthetic_dataset

def dice_loss(pred, target, num_classes=3, smooth=1e-6):
    """
    Calculates Dice Loss for multi-class segmentation.
    pred: [B, C, H, W] logits
    target: [B, H, W] labels
    """
    pred_softmax = F.softmax(pred, dim=1)
    target_one_hot = F.one_hot(target, num_classes=num_classes).permute(0, 3, 1, 2).float()
    
    # Sum over batch, height, width
    intersection = (pred_softmax * target_one_hot).sum(dim=(0, 2, 3))
    total = (pred_softmax + target_one_hot).sum(dim=(0, 2, 3))
    
    dice = (2. * intersection + smooth) / (total + smooth)
    # Return 1 - mean Dice Coefficient
    return 1.0 - dice.mean()

def calculate_metrics(pred_logits, target, num_classes=3):
    """
    Calculates overall Pixel Accuracy and IoU per class.
    pred_logits: [B, C, H, W]
    target: [B, H, W]
    """
    pred_mask = torch.argmax(pred_logits, dim=1) # [B, H, W]
    
    # Pixel Accuracy
    correct = (pred_mask == target).sum().item()
    total_pixels = target.numel()
    pixel_acc = correct / total_pixels
    
    # Class IoU
    class_ious = []
    for c in range(num_classes):
        pred_c = (pred_mask == c)
        target_c = (target == c)
        
        intersection = (pred_c & target_c).sum().item()
        union = (pred_c | target_c).sum().item()
        
        if union == 0:
            class_ious.append(1.0) # Treat as 100% agreement if class is completely absent in both
        else:
            class_ious.append(intersection / union)
            
    return pixel_acc, class_ious

def train_one_epoch(model, dataloader, optimizer, criterion, device, num_classes=3):
    model.train()
    running_loss = 0.0
    total_acc = 0.0
    all_ious = []
    
    for images, masks in dataloader:
        images = images.to(device)
        masks = masks.to(device)
        
        optimizer.zero_grad()
        
        outputs = model(images)
        
        # Loss: CrossEntropy + Dice
        ce_loss = criterion(outputs, masks)
        d_loss = dice_loss(outputs, masks, num_classes=num_classes)
        loss = ce_loss + 0.5 * d_loss
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
        
        # Metric calculation
        acc, ious = calculate_metrics(outputs, masks, num_classes)
        total_acc += acc * images.size(0)
        all_ious.append(ious)
        
    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = total_acc / len(dataloader.dataset)
    epoch_miou = np.mean(all_ious)
    
    return epoch_loss, epoch_acc, epoch_miou

@torch.no_grad()
def validate(model, dataloader, criterion, device, num_classes=3):
    model.eval()
    running_loss = 0.0
    total_acc = 0.0
    all_ious = []
    
    for images, masks in dataloader:
        images = images.to(device)
        masks = masks.to(device)
        
        outputs = model(images)
        
        ce_loss = criterion(outputs, masks)
        d_loss = dice_loss(outputs, masks, num_classes=num_classes)
        loss = ce_loss + 0.5 * d_loss
        
        running_loss += loss.item() * images.size(0)
        
        acc, ious = calculate_metrics(outputs, masks, num_classes)
        total_acc += acc * images.size(0)
        all_ious.append(ious)
        
    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = total_acc / len(dataloader.dataset)
    epoch_miou = np.mean(all_ious)
    
    return epoch_loss, epoch_acc, epoch_miou

def plot_history(history, output_dir):
    epochs = range(1, len(history['train_loss']) + 1)
    
    plt.figure(figsize=(12, 5))
    
    # Loss plot
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], 'b-', label='Train Loss')
    plt.plot(epochs, history['val_loss'], 'r-', label='Val Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Accuracy / mIoU plot
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['train_miou'], 'b-', label='Train mIoU')
    plt.plot(epochs, history['val_miou'], 'r-', label='Val mIoU')
    plt.plot(epochs, history['val_acc'], 'g--', label='Val Pixel Acc')
    plt.title('Training Metrics')
    plt.xlabel('Epochs')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'loss_curve.png'))
    plt.close()
    print(f"Training plots saved to: {os.path.join(output_dir, 'loss_curve.png')}")

def main():
    parser = argparse.ArgumentParser(description="Train U-Net on Satellite Land Cover Imagery")
    parser.add_argument('--data_dir', type=str, default='ml/data', help='Path to dataset directory')
    parser.add_argument('--epochs', type=str, default='15', help='Number of epochs (can be integer)')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--generate_data', action='store_true', help='Force generation of synthetic data')
    parser.add_argument('--num_samples', type=int, default=100, help='Number of synthetic samples to generate')
    
    args = parser.parse_args()
    
    # Handle epochs parsing since command line arguments can be input in various ways
    try:
        epochs = int(args.epochs)
    except ValueError:
        epochs = 15

    # Check/generate data
    train_images_path = os.path.join(args.data_dir, 'train', 'images')
    if args.generate_data or not os.path.exists(train_images_path) or len(os.listdir(train_images_path)) == 0:
        print("Dataset not found or generator forced. Generating synthetic satellite imagery...")
        generate_synthetic_dataset(args.data_dir, num_images=args.num_samples)
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create transforms
    train_transform = JointTransform(train=True)
    val_transform = JointTransform(train=False)
    
    # Datasets & Dataloaders
    train_dataset = SatelliteDataset(os.path.join(args.data_dir, 'train'), transform=train_transform)
    val_dataset = SatelliteDataset(os.path.join(args.data_dir, 'val'), transform=val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    print(f"Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")
    
    # Initialize Model, Optimizer, and Loss
    model = UNet(n_channels=3, n_classes=3).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    best_val_loss = float('inf')
    best_val_miou = 0.0
    
    history = {
        'train_loss': [], 'train_miou': [],
        'val_loss': [], 'val_miou': [], 'val_acc': []
    }
    
    checkpoint_dir = 'ml/checkpoints'
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    print("Starting Training Loop...")
    for epoch in range(1, epochs + 1):
        train_loss, train_acc, train_miou = train_one_epoch(
            model, train_loader, optimizer, criterion, device, num_classes=3
        )
        val_loss, val_acc, val_miou = validate(
            model, val_loader, criterion, device, num_classes=3
        )
        
        history['train_loss'].append(train_loss)
        history['train_miou'].append(train_miou)
        history['val_loss'].append(val_loss)
        history['val_miou'].append(val_miou)
        history['val_acc'].append(val_acc)
        
        print(f"Epoch {epoch:02d}/{epochs:02d} | "
              f"Train Loss: {train_loss:.4f} - mIoU: {train_miou:.4f} | "
              f"Val Loss: {val_loss:.4f} - mIoU: {val_miou:.4f} - Acc: {val_acc:.4f}")
              
        # Save checkpoints
        if val_miou > best_val_miou:
            best_val_miou = val_miou
            checkpoint_path = os.path.join(checkpoint_dir, 'best_model.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_miou': val_miou,
                'val_loss': val_loss
            }, checkpoint_path)
            print(f"==> Saved New Best Model Checkpoint to: {checkpoint_path} (Val mIoU: {val_miou:.4f})")
            
    # Save training curves
    plot_history(history, 'ml')
    print("Training finished!")

if __name__ == '__main__':
    main()
