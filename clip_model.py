import os, sys, pickle, random
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPProcessor, CLIPModel, CLIPTokenizer

# --- Custom Dataset ---
class CLIPDataset(Dataset):
    def __init__(self, mode='train', transform=None):
        # Load the train-test splits (assumed to be in 'train_test_paths.pickle')
        splits = pickle.load(open('train_test_paths.pickle', 'rb'))
        if mode == 'train':
            self.img_paths = splits['train_imgs']
        elif mode == 'val':
            self.img_paths = splits['test_imgs']
        else:
            raise ValueError("mode must be 'train' or 'val'")
        
        # Load image labels from paths (assume label is determined by presence of '/left/' or '/right/' in the path)
        self.labels = []
        for path in self.img_paths:
            if '/left/' in path:
                self.labels.append(0)
            elif '/right/' in path:
                self.labels.append(1)
            else:
                raise ValueError("Unknown label for image: " + path)
        
        # Load paired article texts.
        # NOTE: This file should be a dictionary mapping image paths to raw text.
        self.article_texts = pickle.load(open('article_texts.pickle', 'rb'))
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        path = self.img_paths[idx]
        label = self.labels[idx]
        # Load image (retry if needed)
        while True:
            try:
                img = Image.open(path).convert('RGB')
                break
            except Exception as e:
                idx = random.randint(0, len(self.img_paths) - 1)
                path = self.img_paths[idx]
                label = self.labels[idx]
                continue
        
        if self.transform:
            img = self.transform(img)
        # Get the raw text for the image
        text = self.article_texts.get(path, " ")  # default to empty string if missing
        
        return path, img, label, text

# --- Model Definition ---
class CLIPBiasClassifier(nn.Module):
    def __init__(self, clip_model_name='openai/clip-vit-base-patch32', num_classes=2):
        super().__init__()
        # Load pre-trained CLIP model
        self.clip = CLIPModel.from_pretrained(clip_model_name)
        # We will use CLIP's processor (which wraps the tokenizer and image transforms)
        self.processor = CLIPProcessor.from_pretrained(clip_model_name)
        # Determine dimensions of image and text embeddings
        vision_dim = self.clip.config.vision_config.hidden_size
        text_dim = self.clip.config.text_config.hidden_size
        fusion_dim = vision_dim + text_dim
        # A simple fusion classification head
        self.classifier = nn.Linear(fusion_dim, num_classes)
    
    def forward(self, images, texts):
        # Use CLIP to get image and text features
        # texts is a dict with keys 'input_ids' and 'attention_mask'
        image_features = self.clip.get_image_features(pixel_values=images)  # (batch, vision_dim)
        text_features = self.clip.get_text_features(input_ids=texts['input_ids'],
                                                     attention_mask=texts['attention_mask'])  # (batch, text_dim)
        # Optionally normalize
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        # Concatenate the two modalities
        fused = torch.cat([image_features, text_features], dim=1)
        logits = self.classifier(fused)
        return logits

# --- Training Loop ---
def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Define image transforms for CLIP; note: CLIPProcessor applies its own transforms
    # so you can choose to use the processor for both modalities.
    # Here we let the processor handle image preprocessing.
    # Create dataset and dataloader
    train_dataset = CLIPDataset(mode='train')
    val_dataset = CLIPDataset(mode='val')
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=8)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4)
    
    model = CLIPBiasClassifier().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    num_epochs = 10  # Adjust as needed

    # We'll use the CLIPProcessor for tokenization and image normalization.
    processor = model.processor

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        total = 0
        correct = 0

        for batch in train_loader:
            paths, images, labels, texts = batch
            # Process images and texts using CLIP's processor
            inputs = processor(text=texts, images=images, return_tensors="pt", padding=True)
            # Move inputs to device
            pixel_values = inputs['pixel_values'].to(device)
            # The text tensors are already padded.
            text_inputs = {
                'input_ids': inputs['input_ids'].to(device),
                'attention_mask': inputs['attention_mask'].to(device)
            }
            labels = torch.tensor(labels, dtype=torch.long).to(device)

            optimizer.zero_grad()
            logits = model(pixel_values, text_inputs)
            loss = F.cross_entropy(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * labels.size(0)
            total += labels.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
        
        train_loss = running_loss / total
        train_acc = correct / total

        # Evaluate on validation set
        model.eval()
        val_loss = 0.0
        val_total = 0
        val_correct = 0
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for batch in val_loader:
                paths, images, labels, texts = batch
                inputs = processor(text=texts, images=images, return_tensors="pt", padding=True)
                pixel_values = inputs['pixel_values'].to(device)
                text_inputs = {
                    'input_ids': inputs['input_ids'].to(device),
                    'attention_mask': inputs['attention_mask'].to(device)
                }
                labels = torch.tensor(labels, dtype=torch.long).to(device)
                logits = model(pixel_values, text_inputs)
                loss = F.cross_entropy(logits, labels)
                val_loss += loss.item() * labels.size(0)
                val_total += labels.size(0)
                preds = logits.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                all_preds.extend(preds.cpu().numpy().tolist())
                all_labels.extend(labels.cpu().numpy().tolist())
        
        val_loss /= val_total
        val_acc = val_correct / val_total
        print(f"Epoch {epoch+1}/{num_epochs} -- Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        # Optionally: Save model checkpoint
        torch.save(model.state_dict(), f'clip_bias_model_epoch{epoch+1}.pth')

if __name__ == '__main__':
    train()
