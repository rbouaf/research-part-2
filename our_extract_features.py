import numpy as np
import pickle, re
import os
import sys
import json
import pdb, random, glob, gzip
import time
import traceback
import signal
import torch
from torchvision import transforms, models
from enum import Enum
from PIL import Image
import torch.nn.functional as F
from tqdm import tqdm
from tensorboardX import SummaryWriter
import sklearn.utils
from sklearn.metrics import classification_report

class MyDataset(torch.utils.data.Dataset):
    def __init__(self):
        super().__init__()
        paths = pickle.load(open('train_test_paths.pickle', 'rb'))
        # We combine the training and testing images so that evaluation runs over all images.
        paths = [pth for pth in paths['train_imgs']+paths['test_imgs']]
        self.dataset = paths
        self.transform = transforms.Compose([
            transforms.Resize(size=(224,224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        while True:
            try:
                img = Image.open(self.dataset[index]).convert('RGB')
                break
            except:
                index = random.randrange(len(self.dataset))  # randomly choose a different image since this one failed
                continue
        return self.transform(img), self.dataset[index]

def weight_init(m):
    if isinstance(m, (torch.nn.Linear, torch.nn.Conv2d)):
        torch.nn.init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias.data, 0.1)
    elif isinstance(m, (torch.nn.BatchNorm2d, torch.nn.BatchNorm1d)):
        m.weight.data.uniform_()
        m.bias.data.zero_()

class ImageModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet50 = models.resnet50(pretrained=True)
        self.resnet50 = torch.nn.Sequential(*list(self.resnet50.children())[:-1])
    def forward(self, x):
        x = self.resnet50(x)
        x = x.view(x.shape[0], -1)  # flatten, preserving batch dim
        return x

def main():
    # Create a directory for outputs if it does not exist.
    os.makedirs('classifiers', exist_ok=True)

    # Set up the test DataLoader.
    test_dataloader = torch.utils.data.DataLoader(
        dataset=MyDataset(),
        batch_size=32,
        shuffle=False,
        num_workers=8
    )
    
    # Initialize model and load state.
    img_model = ImageModel().cuda().eval()
    # You pass the model path as command-line argument.
    img_model.load_state_dict(torch.load(sys.argv[1]), strict=False)
    img_model = torch.nn.DataParallel(img_model).cuda()

    feature_dict = {}
    checkpoint_file = 'stage_1_features_checkpoint.pickle'
    
    # If a checkpoint exists, load previously computed features.
    if os.path.exists(checkpoint_file):
        print("Loading checkpoint from", checkpoint_file)
        feature_dict = pickle.load(open(checkpoint_file, 'rb'))
    else:
        print("No checkpoint found; starting from scratch.")
    
    # Optional: set up a SIGTERM handler to save a checkpoint if the job is terminated.
    def handle_sigterm(signum, frame):
        print("SIGTERM received, saving checkpoint before exit...")
        pickle.dump(feature_dict, open(checkpoint_file, 'wb'))
        sys.exit(0)
    signal.signal(signal.SIGTERM, handle_sigterm)
    
    # Evaluation loop with checkpointing every 100 batches.
    batch_counter = 0
    with tqdm(total=len(test_dataloader), ascii=True, leave=False, desc='eval') as pbar:
        for i, (images, paths) in enumerate(test_dataloader):
            # Check if each image has already been processed.
            # (You can also filter the batch to avoid reprocessing images.)
            images = images.float().cuda()
            predicted_features = img_model(images).cpu().data.numpy()
            for j, path in enumerate(paths):
                if path not in feature_dict:
                    feature_dict[path] = predicted_features[j, :]
            batch_counter += 1
            # Save checkpoint every 100 batches
            if batch_counter % 100 == 0:
                pickle.dump(feature_dict, open(checkpoint_file, 'wb'))
                print(f"Checkpoint saved after batch {batch_counter}")
            pbar.update(1)
    
    # Final save of the feature dictionary.
    pickle.dump(feature_dict, open('stage_1_features.pickle', 'wb'))
    print("Evaluation complete. Final features saved to 'stage_1_features.pickle'.")
    

if __name__ == '__main__':
    main()
