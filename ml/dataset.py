import os
import random
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

class SatelliteDataset(Dataset):
    """
    Custom Dataset for loading satellite images and segmentation masks.
    """
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        
        self.images_dir = os.path.join(data_dir, 'images')
        self.masks_dir = os.path.join(data_dir, 'masks')
        
        if not os.path.exists(self.images_dir):
            raise FileNotFoundError(f"Images directory not found at: {self.images_dir}")
            
        self.image_filenames = sorted([
            f for f in os.listdir(self.images_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))
        ])

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        img_name = self.image_filenames[idx]
        img_path = os.path.join(self.images_dir, img_name)
        mask_path = os.path.join(self.masks_dir, img_name) # Assuming mask has same filename
        
        # Load image (RGB)
        image = Image.open(img_path).convert('RGB')
        # Load mask (Grayscale / integer labels)
        mask = Image.open(mask_path).convert('L')
        
        # Apply transforms
        if self.transform:
            image, mask = self.transform(image, mask)
        else:
            # Default fallback to tensors
            image = TF.to_tensor(image)
            # Mask should be long tensor without channel dimension for CrossEntropyLoss
            mask = torch.from_numpy(np.array(mask)).long()
            
        return image, mask

# Dataset transform utility for joint augmentations
class JointTransform:
    def __init__(self, train=True, image_size=(256, 256)):
        self.train = train
        self.image_size = image_size

    def __call__(self, image, mask):
        # Resize
        image = TF.resize(image, self.image_size)
        mask = TF.resize(mask, self.image_size, interpolation=TF.InterpolationMode.NEAREST)
        
        if self.train:
            # Random horizontal flipping
            if random.random() > 0.5:
                image = TF.hflip(image)
                mask = TF.hflip(mask)
                
            # Random vertical flipping
            if random.random() > 0.5:
                image = TF.vflip(image)
                mask = TF.vflip(mask)
                
            # Random rotation (90, 180, 270 degrees)
            angle = random.choice([0, 90, 180, 270])
            if angle > 0:
                image = TF.rotate(image, angle)
                mask = TF.rotate(mask, angle)

        # Transform to tensor
        image = TF.to_tensor(image)
        # Normalize image: standard ImageNet stats or simple 0.5 mean/std
        image = TF.normalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
        # Mask conversion to long tensor
        mask_np = np.array(mask)
        mask = torch.from_numpy(mask_np).long()
        
        return image, mask


def generate_synthetic_image_pair(width=256, height=256):
    """
    Procedurally generates a synthetic satellite image (RGB) and its corresponding 
    3-class segmentation mask.
    Classes:
      0: Built-up (Urban) - Gray/Reddish
      1: Vegetation - Green
      2: Water - Blue
    """
    # Create blank canvases
    img = Image.new('RGB', (width, height))
    mask = Image.new('L', (width, height), color=1) # Default to 1 (Vegetation)
    
    draw_img = ImageDraw.Draw(img)
    draw_mask = ImageDraw.Draw(mask)
    
    # 1. Background / Vegetation texture
    # Fill base vegetation color
    base_veg_r = random.randint(35, 65)
    base_veg_g = random.randint(110, 160)
    base_veg_b = random.randint(35, 65)
    draw_img.rectangle([0, 0, width, height], fill=(base_veg_r, base_veg_g, base_veg_b))
    
    # Add random patches of forest/grassland (different greens)
    for _ in range(15):
        cx = random.randint(0, width)
        cy = random.randint(0, height)
        radius = random.randint(20, 80)
        veg_r = max(10, base_veg_r + random.randint(-20, 10))
        veg_g = max(60, base_veg_g + random.randint(-40, 20))
        veg_b = max(10, base_veg_b + random.randint(-20, 10))
        
        # Draw soft blobs for texture (using alpha blended ellipses on RGB)
        overlay = Image.new('RGBA', (width, height), (0, 0, 0, 0))
        overlay_draw = ImageDraw.Draw(overlay)
        overlay_draw.ellipse([cx - radius, cy - radius, cx + radius, cy + radius], 
                             fill=(veg_r, veg_g, veg_b, 100))
        img.paste(overlay, (0, 0), overlay)

    # 2. Water (Class 2) - River or Lake
    if random.random() > 0.3:
        # River (bezier curve style or multi-segment line)
        river_points = []
        start_x = random.choice([0, width])
        start_y = random.randint(0, height)
        river_points.append((start_x, start_y))
        
        mid_x = random.randint(width // 4, 3 * width // 4)
        mid_y = random.randint(height // 4, 3 * height // 4)
        river_points.append((mid_x, mid_y))
        
        end_x = width - start_x
        end_y = random.randint(0, height)
        river_points.append((end_x, end_y))
        
        river_width = random.randint(10, 25)
        river_color = (random.randint(15, 30), random.randint(60, 100), random.randint(160, 220))
        
        # Interpolate points to draw a smoother curve
        curve_points = []
        for t in np.linspace(0, 1, 20):
            # Quadratic Bezier formula
            px = int((1-t)**2 * river_points[0][0] + 2*(1-t)*t * river_points[1][0] + t**2 * river_points[2][0])
            py = int((1-t)**2 * river_points[0][1] + 2*(1-t)*t * river_points[1][1] + t**2 * river_points[2][1])
            curve_points.append((px, py))
            
        # Draw on image
        draw_img.line(curve_points, fill=river_color, width=river_width, joint='round')
        # Draw on mask
        draw_mask.line(curve_points, fill=2, width=river_width, joint='round')
        
    # 3. Built-up Area (Class 0) - Cities / Roads / Buildings
    # Define an urban center
    num_towns = random.randint(1, 3)
    for _ in range(num_towns):
        town_x = random.randint(width // 6, 5 * width // 6)
        town_y = random.randint(height // 6, 5 * height // 6)
        town_radius = random.randint(40, 80)
        
        # Generate roads extending from town center
        num_roads = random.randint(3, 6)
        for _ in range(num_roads):
            road_angle = random.uniform(0, 2 * np.pi)
            road_length = random.randint(50, 150)
            road_end_x = int(town_x + road_length * np.cos(road_angle))
            road_end_y = int(town_y + road_length * np.sin(road_angle))
            
            road_color = (120, 120, 120)
            road_width = random.choice([3, 5])
            
            draw_img.line([(town_x, town_y), (road_end_x, road_end_y)], fill=road_color, width=road_width)
            draw_mask.line([(town_x, town_y), (road_end_x, road_end_y)], fill=0, width=road_width)
            
            # Place houses along the roads
            num_houses = random.randint(5, 12)
            for i in range(num_houses):
                t = random.uniform(0.1, 0.9)
                hx = int(town_x + t * road_length * np.cos(road_angle))
                hy = int(town_y + t * road_length * np.sin(road_angle))
                
                # Offset house slightly from the road
                offset_dist = random.randint(6, 12)
                offset_angle = road_angle + np.pi/2 * random.choice([-1, 1])
                hx += int(offset_dist * np.cos(offset_angle))
                hy += int(offset_dist * np.sin(offset_angle))
                
                house_sz = random.randint(4, 10)
                # Randomize building color (grey roofs or reddish roofs)
                if random.random() > 0.4:
                    building_color = (random.randint(160, 200), random.randint(70, 100), random.randint(60, 90)) # Red tiles
                else:
                    building_color = (random.randint(100, 140), random.randint(100, 140), random.randint(100, 140)) # Concrete grey
                
                draw_img.rectangle([hx - house_sz, hy - house_sz, hx + house_sz, hy + house_sz], fill=building_color)
                draw_mask.rectangle([hx - house_sz, hy - house_sz, hx + house_sz, hy + house_sz], fill=0)

    # Apply sensor-like noise and blur to simulate satellite capture
    img_np = np.array(img).astype(np.float32)
    # Add Gaussian/uniform noise
    noise = np.random.normal(0, 8, img_np.shape)
    img_np = np.clip(img_np + noise, 0, 255).astype(np.uint8)
    
    img = Image.fromarray(img_np)
    # Apply a tiny blur to soften sharp procedural shapes
    img = img.filter(ImageFilter.GaussianBlur(radius=0.5))
    
    return img, mask


def generate_synthetic_dataset(output_dir, num_images=100, image_size=(256, 256)):
    """
    Generates a full synthetic satellite dataset split into train and val.
    """
    train_dir = os.path.join(output_dir, 'train')
    val_dir = os.path.join(output_dir, 'val')
    
    for split_dir in [train_dir, val_dir]:
        os.makedirs(os.path.join(split_dir, 'images'), exist_ok=True)
        os.makedirs(os.path.join(split_dir, 'masks'), exist_ok=True)
        
    num_train = int(num_images * 0.8)
    
    print(f"Generating synthetic dataset in '{output_dir}'...")
    print(f"Generating {num_train} training samples...")
    for idx in range(num_train):
        img, mask = generate_synthetic_image_pair(image_size[0], image_size[1])
        img.save(os.path.join(train_dir, 'images', f'sat_{idx:04d}.png'))
        mask.save(os.path.join(train_dir, 'masks', f'sat_{idx:04d}.png'))
        
    num_val = num_images - num_train
    print(f"Generating {num_val} validation samples...")
    for idx in range(num_val):
        img, mask = generate_synthetic_image_pair(image_size[0], image_size[1])
        img.save(os.path.join(val_dir, 'images', f'sat_{idx:04d}.png'))
        mask.save(os.path.join(val_dir, 'masks', f'sat_{idx:04d}.png'))
        
    print("Synthetic dataset generation completed successfully!")

if __name__ == '__main__':
    # Test generation of a single pair
    img, mask = generate_synthetic_image_pair()
    print("Single image generated:")
    print("- Image Size:", img.size)
    print("- Mask Size:", mask.size)
    print("- Unique classes in mask:", np.unique(np.array(mask)))
    
    # Test dataset generator
    test_dir = 'temp_dataset'
    generate_synthetic_dataset(test_dir, num_images=10)
    
    # Load with SatelliteDataset
    train_dataset = SatelliteDataset(os.path.join(test_dir, 'train'), transform=JointTransform(train=True))
    print(f"Loaded dataset length: {len(train_dataset)}")
    x, y = train_dataset[0]
    print("Tensor Image shape:", x.shape)
    print("Tensor Mask shape:", y.shape)
    
    # Clean up temp dataset directory
    import shutil
    shutil.rmtree(test_dir)
    print("Dataset module self-test passed!")
