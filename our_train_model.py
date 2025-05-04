import numpy as np
import pickle, re
import os
import sys
import json
import pdb, random, glob, gzip
import time
import traceback
import torch
from torchvision import transforms, models
from enum import Enum
from PIL import Image
import torch.nn.functional as F
from tqdm import tqdm
from tensorboardX import SummaryWriter
import sklearn.utils
class MyDataset(torch.utils.data.Dataset):
    all_imgs_and_labels = []
    class_weights = []
    all_imgs_to_annotations = {}
    add_dim = 0
    def __init__(self, mode):
        super().__init__()
        if not MyDataset.all_imgs_and_labels:
            MyDataset.all_imgs_to_annotations = pickle.load(open('doc2vec_features.pickle', 'rb'))
            MyDataset.add_dim = list(MyDataset.all_imgs_to_annotations.values())[0].size
            paths = pickle.load(open('train_test_paths.pickle', 'rb'))['train_imgs']
            for path in paths:
                if '/left/' in path:
                    MyDataset.all_imgs_and_labels.append((path, 0))
                elif '/right/' in path:
                    MyDataset.all_imgs_and_labels.append((path, 1))
            MyDataset.class_weights = torch.from_numpy(sklearn.utils.compute_class_weight('balanced', np.array([0,1]), [y for _,y in MyDataset.all_imgs_and_labels])).float().cuda()
            random.shuffle(MyDataset.all_imgs_and_labels)
            MyDataset.all_imgs_and_labels = dict(MyDataset.all_imgs_and_labels)
        train_imgs = list(MyDataset.all_imgs_and_labels.keys())[:int(round(0.90 * len(MyDataset.all_imgs_and_labels.keys())))]
        val_imgs = list(MyDataset.all_imgs_and_labels.keys())[int(round(0.90 * len(MyDataset.all_imgs_and_labels.keys()))):]
        if mode=='train':
            self.dataset=train_imgs
            self.transform = transforms.Compose([
                transforms.Resize(size=(224,224)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])])
        elif mode=='val':
            self.dataset=val_imgs
            self.transform = transforms.Compose([
                transforms.Resize(size=(224,224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])])
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, index):
        while True:
            try:
                img = Image.open(self.dataset[index]).convert('RGB')
                break
            except:
                index = random.randrange(len(self.dataset)) # randomly choose a different image since this one failed
                continue
        return self.transform(img), MyDataset.all_imgs_and_labels[self.dataset[index]], MyDataset.all_imgs_to_annotations[self.dataset[index]]
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
        self.fusion = torch.nn.Linear(2048 + MyDataset.add_dim, 512, bias=True)
        self.fusion.apply(weight_init)
        self.classifier = torch.nn.Linear(512, 2, bias=True)
        self.classifier.apply(weight_init)
    def forward(self, x, y):
        x = self.resnet50(x)
        x = x.view(x.shape[0], -1) # flatten, preserving batch dim
        x = torch.cat([x, y], dim=1)
        x = self.fusion(x)
        x = self.classifier(x)
        return x
def main():
    print("cuDNN enabled:", torch.backends.cudnn.enabled)
    print("CUDA:", torch.version.cuda)
    print("cuDNN:", torch.backends.cudnn.version())
    print("GPU:", torch.cuda.get_device_name(0))

    train_dataloader = torch.utils.data.DataLoader(
        dataset=MyDataset('train'),
        batch_size=16,
        shuffle=True,
        num_workers=4
    )
    test_dataloader = torch.utils.data.DataLoader(
        dataset=MyDataset('val'),
        batch_size=10,
        shuffle=False,
        num_workers=4
    )
    writer = SummaryWriter('classifiers2/')
    img_model = torch.nn.DataParallel(ImageModel()).cuda()
    for p in img_model.parameters():
        p.requires_grad = True
    optimizer = torch.optim.Adam(list(img_model.parameters()), lr=0.0001)
    itr = 0

    # Set up checkpointing
    checkpoint_path = 'classifiers2/checkpoint_latest.pth'
    start_epoch = 1
    if os.path.exists(checkpoint_path):
        print("Loading checkpoint from", checkpoint_path)
        checkpoint = torch.load(checkpoint_path)
        img_model.module.load_state_dict(checkpoint['model_state'])
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        start_epoch = checkpoint.get('epoch', 1) + 1
        itr = checkpoint.get('itr', 0)

    # Use a global variable to store current epoch (for the signal handler)
    global_epoch = start_epoch

    # Signal handler to catch SLURM termination (SIGTERM) and save a checkpoint
    import signal
    def handle_sigterm(signum, frame):
        print("SIGTERM received, saving checkpoint before exit...")
        checkpoint_data = {
            'epoch': global_epoch,
            'itr': itr,
            'model_state': img_model.module.state_dict(),
            'optimizer_state': optimizer.state_dict()
        }
        torch.save(checkpoint_data, checkpoint_path)
        sys.exit(0)
    signal.signal(signal.SIGTERM, handle_sigterm)

    # Training loop with checkpointing every 1000 iterations
    for e in tqdm(range(start_epoch, 50), ascii=True, desc='Epoch'):
        global_epoch = e  # update global epoch so the signal handler uses the current epoch
        img_model.train()
        with tqdm(total=len(train_dataloader), ascii=True, leave=False, desc='iter') as pbar:
            for i, (images, labels, semantics) in enumerate(train_dataloader):
                itr += 1
                optimizer.zero_grad()
                images = images.float().cuda()
                labels = labels.long().cuda()
                semantics = semantics.float().cuda()
                classifications = img_model(images, semantics)
                loss = F.cross_entropy(input=classifications, target=labels, weight=MyDataset.class_weights)
                loss.backward()
                optimizer.step()
                writer.add_scalar('data/train_loss', loss.item(), itr)
                pbar.update(1)
                
                # Save checkpoint every 1000 iterations
                if itr % 1000 == 0:
                    checkpoint_data = {
                        'epoch': e,
                        'itr': itr,
                        'model_state': img_model.module.state_dict(),
                        'optimizer_state': optimizer.state_dict()
                    }
                    torch.save(checkpoint_data, checkpoint_path)
                    print(f"Checkpoint saved at epoch {e}, iteration {itr}")

        img_model.eval()
        losses = []
        with tqdm(total=len(test_dataloader), ascii=True, leave=False, desc='eval') as pbar:
            for i, (images, labels, semantics) in enumerate(test_dataloader):
                images = images.float().cuda()
                labels = labels.long().cuda()
                semantics = semantics.float().cuda()
                classifications = img_model(images, semantics)
                loss = F.cross_entropy(input=classifications, target=labels)
                losses.append(loss.item())
                pbar.update(1)
        writer.add_scalar('data/val_loss', sum(losses) / len(losses), e)
        
        # Save a checkpoint at the end of the epoch
        checkpoint_data = {
            'epoch': e,
            'itr': itr,
            'model_state': img_model.module.state_dict(),
            'optimizer_state': optimizer.state_dict()
        }
        torch.save(checkpoint_data, checkpoint_path)
        print(f"Epoch {e} completed, checkpoint saved.")
        
        # Also save a model snapshot per epoch
        try:
            torch.save(img_model.module.state_dict(), f'classifiers2/img_model_{e}.pth')
        except Exception as ex:
            print('Failed saving model snapshot:', ex)
            continue

if __name__ == '__main__':
    main()
