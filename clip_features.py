import os, sys, pickle, random
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPProcessor, CLIPModel

# A simple dataset that loads images (no text needed)
class ImageOnlyDataset(Dataset):
    def __init__(self, mode='train'):
        splits = pickle.load(open('train_test_paths.pickle', 'rb'))
        if mode == 'train':
            self.img_paths = splits['train_imgs']
        elif mode == 'val':
            self.img_paths = splits['test_imgs']
        else:
            raise ValueError("mode must be 'train' or 'val'")
    
    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, idx):
        path = self.img_paths[idx]
        while True:
            try:
                img = Image.open(path).convert('RGB')
                break
            except Exception:
                idx = random.randint(0, len(self.img_paths) - 1)
                path = self.img_paths[idx]
                continue
        return path, img

def extract_features(model, processor, dataloader, device):
    model.eval()
    feature_dict = {}
    with torch.no_grad():
        for batch in dataloader:
            paths, images = batch
            inputs = processor(images=images, return_tensors="pt", padding=True)
            pixel_values = inputs['pixel_values'].to(device)
            # Get image features from the CLIP image encoder
            image_features = model.get_image_features(pixel_values=pixel_values)
            image_features = image_features.cpu().numpy()
            for i, path in enumerate(paths):
                feature_dict[path] = image_features[i]
    return feature_dict

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Load the fine-tuned CLIP model from stage 1 checkpoint (choose the best epoch)
    model = CLIPModel.from_pretrained('openai/clip-vit-base-patch32')
    model.load_state_dict(torch.load('clip_bias_model_epoch10.pth', map_location=device))
    model.to(device)
    processor = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')
    
    # Create dataset and dataloader for both train and validation sets
    train_dataset = ImageOnlyDataset(mode='train')
    val_dataset = ImageOnlyDataset(mode='val')
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
    
    # Extract and save features
    train_features = extract_features(model, processor, train_loader, device)
    val_features = extract_features(model, processor, val_loader, device)
    with open('stage_1_clip_features_train.pickle', 'wb') as f:
        pickle.dump(train_features, f)
    with open('stage_1_clip_features_val.pickle', 'wb') as f:
        pickle.dump(val_features, f)
    print("Feature extraction complete.")

if __name__ == '__main__':
    main()
