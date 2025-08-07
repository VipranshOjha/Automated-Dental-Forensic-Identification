import os
import json
import io
import random
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import gc

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision import models
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.model_selection import KFold
from collections import defaultdict # For easier grouping by diagnosis

# --- Diagnosis Mapping: Fine to Broad Category ---
diagnosis_mapping = {
    # Caries
    "caries, mild": "caries",
    "caries, moderate": "caries",
    "caries, severe": "caries",

    # Periapical Lesion
    "periapical lesion": "periapical lesion",

    # Restoration
    "restoration, amalgam": "restoration",
    "restoration, composite": "restoration",

    # Crown
    "crown, full coverage": "crown",
    "crown, partial coverage": "crown",

    # Calculus
    "calculus, mild": "calculus",
    "calculus, severe": "calculus",

    # Root Canal Filling
    "root canal filling, short": "root canal filling",
    "root canal filling, adequate": "root canal filling",
    "root canal filling, overfill": "root canal filling",

    # Add other mappings as needed
    # "impacted_tooth": "impacted_tooth",
}

# Enable mixed precision training
torch.backends.cudnn.benchmark = True # Improves performance for consistent input sizes

# Global transform for image preprocessing
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

class LazySiameseDataset(Dataset):
    def __init__(self, image_paths_and_info, target_size=(256, 256), num_pairs_per_epoch=None,
                 balance_pairs=True, max_neg_ratio=1.0):
        """
        Args:
            image_paths_and_info (list): A list of dictionaries, where each dict
                                        contains {"path": "image/path.jpg", "diagnoses": set_of_diagnosis_ids}. 
            target_size (tuple): Target size for image resizing.
            num_pairs_per_epoch (int): The total number of pairs to generate for each epoch.
                                       If None, generates pairs from a reasonable default.
            balance_pairs (bool): If True, attempts to balance positive and negative pairs.
            max_neg_ratio (float): If balance_pairs is True, this sets the max ratio of negative to positive pairs.
        """
        self.image_paths_and_info = image_paths_and_info
        self.target_size = target_size
        self.balance_pairs = balance_pairs
        self.max_neg_ratio = max_neg_ratio

        self.transform = transforms.Compose([
            transforms.Resize(self.target_size),
            transforms.ToTensor()
        ])

        # Group images by diagnosis for efficient positive pair generation
        self.diagnoses_to_image_indices = defaultdict(list)
        all_image_ids = set() # To store unique image IDs for quick lookup (if needed for patient IDs)
        for i, item in enumerate(self.image_paths_and_info):
            # Each image can have multiple diagnoses
            if 'diagnoses' in item and item['diagnoses']: # Make sure 'diagnoses' key exists and is not empty
                for diagnosis_id in item['diagnoses']:
                    self.diagnoses_to_image_indices[diagnosis_id].append(i)
            # Collect all image_ids for negative pair generation
            all_image_ids.add(i)
        self.all_image_indices = list(all_image_ids) # Convert to list for random.sample

        self._pre_generate_pair_indices(num_pairs_per_epoch)

    def _pre_generate_pair_indices(self, num_pairs_per_epoch):
        """
        Pre-generates the indices of the images that will form pairs,
        attempting to balance positive and negative samples based on diagnosis.
        """
        self.pair_indices_and_labels = [] # Stores (idx1, idx2, label)

        if num_pairs_per_epoch is None:
            # Default to a reasonable number, e.g., 2-3x the number of images
            num_pairs_per_epoch = len(self.image_paths_and_info) * 2

        num_positive_pairs = 0
        num_negative_pairs = 0

        # Prioritize positive pair generation
        # Loop to attempt to fill up to num_pairs_per_epoch for positive pairs
        for _ in range(num_pairs_per_epoch): 
            if self.balance_pairs and num_positive_pairs < num_pairs_per_epoch / 2: # Aim for 50% positive initially
                found_positive_pair = False
                attempts = 0
                max_attempts = len(self.image_paths_and_info) * 2 # Don't loop infinitely

                # Filter for diagnoses that actually have more than one image
                eligible_diagnoses = [d_id for d_id, indices in self.diagnoses_to_image_indices.items() if len(indices) >= 2]

                if not eligible_diagnoses:
                    # If no eligible diagnoses, break the loop for positive pair generation
                    print("WARNING: No eligible diagnoses found with >= 2 images. Cannot generate positive pairs.")
                    break

                while not found_positive_pair and attempts < max_attempts:
                    selected_diagnosis = random.choice(eligible_diagnoses)
                    img_indices_for_diag = self.diagnoses_to_image_indices[selected_diagnosis]

                    # This check is redundant if eligible_diagnoses is correctly filtered
                    # but kept for safety.
                    if len(img_indices_for_diag) >= 2:
                        idx1, idx2 = random.sample(img_indices_for_diag, 2)
                        item1_diagnoses = self.image_paths_and_info[idx1].get('diagnoses', set())
                        item2_diagnoses = self.image_paths_and_info[idx2].get('diagnoses', set())

                        common_diagnoses = item1_diagnoses.intersection(item2_diagnoses)
                        # print(f"Attempting positive pair: img1_idx={idx1}, img2_idx={idx2}")
                        # print(f"  Diagnoses 1: {item1_diagnoses}")
                        # print(f"  Diagnoses 2: {item2_diagnoses}")
                        # print(f"  Common: {common_diagnoses}")

                        if common_diagnoses:
                            self.pair_indices_and_labels.append((idx1, idx2, 1))
                            num_positive_pairs += 1
                            found_positive_pair = True
                            # print(f"  --> POSITIVE PAIR FOUND!")
                    attempts += 1
            else:
                # If we have enough positive pairs or balancing is not required, generate negative pairs
                if num_negative_pairs < num_pairs_per_epoch - num_positive_pairs: # Fill remaining with negative pairs
                    idx1, idx2 = random.sample(self.all_image_indices, 2)
                    item1_diagnoses = self.image_paths_and_info[idx1].get('diagnoses', set())
                    item2_diagnoses = self.image_paths_and_info[idx2].get('diagnoses', set())

                    if not item1_diagnoses.intersection(item2_diagnoses):
                        self.pair_indices_and_labels.append((idx1, idx2, 0))
                        num_negative_pairs += 1
                else:
                    break # Stop if we have generated enough total pairs

        # Ensure we always generate *some* pairs to avoid an empty dataset
        # This will be mostly negative if no positive pairs can be made.
        if not self.pair_indices_and_labels:
            print("WARNING: No pairs (positive or negative) could be generated with current dataset and parameters. Generating only negative pairs as a fallback.")
            # Generate at least 'num_pairs_per_epoch' negative pairs as a fallback
            for _ in range(num_pairs_per_epoch):
                idx1, idx2 = random.sample(self.all_image_indices, 2)
                item1_diagnoses = self.image_paths_and_info[idx1].get('diagnoses', set())
                item2_diagnoses = self.image_paths_and_info[idx2].get('diagnoses', set())
                if not item1_diagnoses.intersection(item2_diagnoses):
                    self.pair_indices_and_labels.append((idx1, idx2, 0))
                    num_negative_pairs += 1
                    if len(self.pair_indices_and_labels) >= num_pairs_per_epoch:
                        break # Stop if we reach the desired number of pairs
        
        print(f"Generated {num_positive_pairs} positive pairs and {num_negative_pairs} negative pairs out of a target of {num_pairs_per_epoch} pairs.")
        if num_positive_pairs > 0:
            print(f"  Negative/Positive Ratio: {num_negative_pairs / num_positive_pairs:.2f}")
        else:
            print("  No positive pairs generated, ratio is undefined.")

    def __len__(self):
        return len(self.pair_indices_and_labels)

    def __getitem__(self, idx):
        idx1, idx2, label_val = self.pair_indices_and_labels[idx]

        item1 = self.image_paths_and_info[idx1]
        item2 = self.image_paths_and_info[idx2]

        img1_path = item1["path"]
        img2_path = item2["path"]

        try:
            with Image.open(img1_path) as im1:
                img1 = self.transform(im1.convert("RGB"))
            with Image.open(img2_path) as im2:
                img2 = self.transform(im2.convert("RGB"))
        except Exception as e:
            print(f"Failed to load/convert image pair at idx {idx} (paths: {img1_path}, {img2_path}): {e}")
            img1 = torch.zeros(3, *self.target_size)
            img2 = torch.zeros(3, *self.target_size)

        label = torch.tensor(label_val, dtype=torch.float32)
        return img1, img2, label

class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
        self.base_model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        # Modify the classifier to output the desired embedding size (e.g., 128)
        self.base_model.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(self.base_model.classifier[1].in_features, 128) # Output embedding size
        )

    def forward_once(self, x):
        return self.base_model(x)

    def forward(self, x1, x2):
        embed1 = F.normalize(self.forward_once(x1), p=2, dim=1)
        embed2 = F.normalize(self.forward_once(x2), p=2, dim=1)
        cosine_sim = F.cosine_similarity(embed1, embed2)
        # Return the raw similarity score for BCEWithLogitsLoss
        # A higher value means more similar. BCEWithLogitsLoss expects a high logit for label 1.
        # Since cosine_sim is 1 for identical and -1 for opposite, using cosine_sim directly is best.
        # If your definition of "distance" is 1-cosine_sim, then you'd want 0 for similar (label 1).
        # Let's align with the common BCEWithLogitsLoss expectation: higher logit for positive class.
        # So, if label=1 means "similar", output should be high. If label=0 means "dissimilar", output should be low.
        return cosine_sim # Output is a similarity score, high for similar, low for dissimilar.

def train_model(model, train_loader, val_loader, epochs=10, lr=1e-3, save_path="model.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Use BCEWithLogitsLoss, which expects logits (raw scores)
    criterion = nn.BCEWithLogitsLoss() 
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_loss = float('inf')
    history = {"loss": [], "val_loss": [], "acc": [], "val_acc": []}

    scaler = torch.amp.GradScaler(enabled=torch.cuda.is_available())

    print(f"Training on device: {device}")

    for epoch in range(epochs):
        model.train()
        total_loss, correct, count = 0, 0, 0
        for batch_idx, (img1, img2, label) in enumerate(train_loader):
            img1, img2, label = img1.to(device), img2.to(device), label.to(device)
            optimizer.zero_grad()

            with torch.amp.autocast(device_type=device.type, dtype=torch.float16, enabled=torch.cuda.is_available()):
                output = model(img1, img2).squeeze()
                loss = criterion(output, label)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item() * len(label)
            
            # Apply sigmoid to output (logits) for prediction thresholding
            pred = (torch.sigmoid(output) > 0.5).float() 
            correct += (pred == label).sum().item()
            count += len(label)

            if batch_idx % 100 == 0:
                 print(f"  Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}")
            
            del img1, img2, label, output, loss
            gc.collect()
            if device.type == 'cuda':
                torch.cuda.empty_cache()

        val_loss, val_acc = evaluate_model(model, val_loader, device, criterion)
        train_loss = total_loss / count
        train_acc = correct / count
        history["loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        print(f"Epoch {epoch+1}: Train Loss {train_loss:.4f}, Val Loss {val_loss:.4f}, Acc {train_acc:.4f}, Val Acc {val_acc:.4f}")

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), save_path)
            print(f"  Model saved to {save_path} (Val Loss: {val_loss:.4f})")
        
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    return history

def evaluate_model(model, loader, device, criterion):
    model.eval()
    total_loss, correct, count = 0, 0, 0
    with torch.no_grad():
        for img1, img2, label in loader:
            img1, img2, label = img1.to(device), img2.to(device), label.to(device)
            with torch.amp.autocast(device_type=device.type, dtype=torch.float16, enabled=torch.cuda.is_available()):
                output = model(img1, img2).squeeze()
                loss = criterion(output, label)
            
            total_loss += loss.item() * len(label)
            pred = (torch.sigmoid(output) > 0.5).float()
            correct += (pred == label).sum().item()
            count += len(label)

            del img1, img2, label, output, loss
            gc.collect()
            if device.type == 'cuda':
                torch.cuda.empty_cache()
    return total_loss / count, correct / count

def plot_training_history(history):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history["acc"], label="Train Accuracy")
    plt.plot(history["val_acc"], label="Val Accuracy")
    plt.legend()
    plt.title("Accuracy")

    plt.subplot(1, 2, 2)
    plt.plot(history["loss"], label="Train Loss")
    plt.plot(history["val_loss"], label="Val Loss")
    plt.legend()
    plt.title("Loss")
    plt.show()

def cross_validate_model(dataset_info, n_splits=5, epochs=10, batch_size=32, num_pairs_per_epoch=None, num_workers=0):
    results = []
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    for fold, (train_indices, val_indices) in enumerate(kf.split(dataset_info)):
        print(f"\nFold {fold + 1}/{n_splits}")

        train_image_paths_and_ids = [dataset_info[i] for i in train_indices]
        val_image_paths_and_ids = [dataset_info[i] for i in val_indices]

        train_siamese_dataset = LazySiameseDataset(train_image_paths_and_ids, num_pairs_per_epoch=num_pairs_per_epoch)
        val_siamese_dataset = LazySiameseDataset(val_image_paths_and_ids, num_pairs_per_epoch=num_pairs_per_epoch // 4 if num_pairs_per_epoch else None)

        train_loader = DataLoader(train_siamese_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
        val_loader = DataLoader(val_siamese_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True)

        model = SiameseNetwork()
        history = train_model(model, train_loader, val_loader, epochs=epochs, save_path=f"siamese_fold_{fold+1}.pth")
        results.append(history)

        del model, train_loader, val_loader, train_siamese_dataset, val_siamese_dataset
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return results

def load_dentex_json_dataset(images_dir, json_path):
    with open(json_path, 'r') as f:
        raw_data = json.load(f)

    image_id_to_filename = {img['id']: img['file_name'] for img in raw_data['images']}
    image_id_to_diagnoses = defaultdict(set)

    for ann in raw_data['annotations']:
        img_id = ann['image_id']
        if 'diagnosis' in ann and ann['diagnosis']:
            diagnoses_raw = []
            if isinstance(ann['diagnosis'], list):
                diagnoses_raw.extend(ann['diagnosis'])
            elif isinstance(ann['diagnosis'], str):
                diagnoses_raw.append(ann['diagnosis'])

            for diag_str in diagnoses_raw:
                normalized_diag = diag_str.strip().lower()
                # Map to broad category if possible, else use normalized string
                broad_category = diagnosis_mapping.get(normalized_diag, normalized_diag)
                image_id_to_diagnoses[img_id].add(broad_category)

    dataset_info = []
    for img_id, filename in image_id_to_filename.items():
        path = os.path.join(images_dir, filename)
        if os.path.exists(path):
            diagnoses_for_image = image_id_to_diagnoses.get(img_id, set())
            dataset_info.append({
                "path": path,
                "image_id": img_id,
                "diagnoses": diagnoses_for_image
            })
        else:
            print(f"Warning: Image file not found at {path}")
    return dataset_info

if __name__ == "__main__":
    # Aggressively clear memory before starting
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Load dataset info (paths and IDs) - this should NOT consume much RAM
    dataset_info = load_dentex_json_dataset(
        "./dentex/training_data/quadrant-enumeration-disease/xrays",
        "./dentex/training_data/quadrant-enumeration-disease/train_quadrant_enumeration_disease.json"
    )
    print(f"Loaded dataset info for {len(dataset_info)} images.")

    # --- IMPORTANT: Adjust these parameters based on your memory ---
    # Start with a very small batch size if you still hit OOM
    # **INCREASE BATCH SIZE**
    current_batch_size = 64 # Try 16, 32, 64. With mixed precision and 8GB VRAM, 64 is often feasible.
    # Number of pairs to generate per epoch for the Siamese dataset.
    # A smaller number means less overhead in the dataset object, but less diversity per epoch.
    # Adjust this based on available RAM and dataset size.
    # For a dataset of 1000 images, 2000 pairs means roughly 2 samples per image.
    current_num_pairs = 4000 # Potentially increase for more diversity per epoch if GPU is well utilized
    current_epochs = 5 # Keep as is for initial testing
    num_dataloader_workers = 4 # **ADD THIS NEW PARAMETER**

    # Run cross-validation
    print("\nStarting Cross-Validation...")
    cv_results = cross_validate_model(
        dataset_info,
        n_splits=3, # Reduced splits for faster testing
        epochs=current_epochs,
        batch_size=current_batch_size,
        num_pairs_per_epoch=current_num_pairs,
        num_workers=num_dataloader_workers # **PASS TO CROSS_VALIDATE_MODEL**
    )

    # Train final model on new split
    print("\nStarting Final Model Training...")
    random.shuffle(dataset_info)
    train_size_final = int(0.8 * len(dataset_info))
    train_paths_ids = dataset_info[:train_size_final]
    test_paths_ids = dataset_info[train_size_final:]

    full_train_siamese_dataset = LazySiameseDataset(train_paths_ids, num_pairs_per_epoch=current_num_pairs * 2)
    full_test_siamese_dataset = LazySiameseDataset(test_paths_ids, num_pairs_per_epoch=current_num_pairs // 2)

    # **UPDATE DATALOADER INSTANTIATION**
    full_train_loader = DataLoader(full_train_siamese_dataset, batch_size=current_batch_size, shuffle=True, num_workers=num_dataloader_workers, pin_memory=True)
    full_test_loader = DataLoader(full_test_siamese_dataset, batch_size=current_batch_size, num_workers=num_dataloader_workers, pin_memory=True)

    final_model = SiameseNetwork()
    final_history = train_model(final_model, full_train_loader, full_test_loader, epochs=current_epochs, save_path="final_model.pth")
    plot_training_history(final_history)

    # Final cleanup
    del full_train_loader, full_test_loader, full_train_siamese_dataset, full_test_siamese_dataset, final_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("Training complete!")