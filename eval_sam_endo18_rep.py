import numpy as np
import torch
import cv2
import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import time

from train_sam_endo18_rep import (
    rep_vit, 
    Model,
    FeatModel,
    LabPicsDataset,
    collate_fn,
    get_positional_encoding
)

rep_vit_model = rep_vit.rep_vit_m1(1024)
model = Model()
feat_model = FeatModel()

def load_models(checkpoint_dir, epoch):
    rep_vit_model.load_state_dict(torch.load(f'{checkpoint_dir}/sam_endo18_rep_rep_{epoch}.torch'))#rep
    model.load_state_dict(torch.load(f'{checkpoint_dir}/sam_endo18_rep_{epoch}.torch'))#
    feat_model.load_state_dict(torch.load(f'{checkpoint_dir}/sam_endo18_rep_feat_{epoch}.torch'))#feat
    return rep_vit_model, model, feat_model

def evaluate(models, dataloader):
    rep_vit_model, model, feat_model = models
    rep_vit_model.eval()
    model.eval()
    feat_model.eval()
    
    iou_list = [[] for _ in range(7)]
    eps = 1e-6
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            image, mask, id = batch
            image = image.to('cuda')
            id = torch.from_numpy(np.array(id)).to(torch.int16).to('cuda')
            
            vit_feat = rep_vit_model(image)
            sparse_feat, dense_feat = feat_model(vit_feat, id)
            prd_masks = model(sparse_feat, dense_feat, id)
            prd_mask = torch.sigmoid(prd_masks)
            gt_mask = torch.tensor(mask[:, 0].astype(np.float32)).cuda()
            
            inter = (gt_mask * (prd_mask > 0.5)).sum(1).sum(1)
            iou = (inter + eps) / (gt_mask.sum(1).sum(1) + (prd_mask > 0.5).sum(1).sum(1) - inter + eps)
            iou = iou.cpu().numpy()
            
            for i, single_id in enumerate(id):
                iou_list[single_id].append(iou[i])
            
            print(f"Processed batch {batch_idx+1}/{len(dataloader)}")
    
    class_iou = [np.mean(iou) if iou else 0 for iou in iou_list]
    mean_iou = np.mean([iou for iou in class_iou if iou > 0])
    
    return class_iou, mean_iou

if __name__ == "__main__":
    for j in range(0,31):
        checkpoint_dir = "./your_ckpt_dir"
        checkpoint_epoch = j
        data_dir = "./your_data_dir"
        batch_size = 1
        num_workers = 8
        
        print("Loading models...")
        models = load_models(checkpoint_dir, checkpoint_epoch)
        for m in models:
            m.to('cuda')
        
        print("Preparing validation data...")
        val_data = []
        sub_folder_list = ['BF', 'CA', 'LND', 'MCS', 'PF', 'SI', 'UP']
        for id, sub_folder in enumerate(sub_folder_list):
            for name in os.listdir(data_dir + "val_gt/" + sub_folder + '/'):
                new_name = name.split('.')[0] + ".png"
                val_data.append({
                    "image": data_dir + "val_og/" + new_name,
                    "annotation": data_dir + "val_gt/" + sub_folder + '/' + name,
                    "id": id
                })
        
        transform = transforms.Compose([
            transforms.Resize((1024, 1024)),
            transforms.ToTensor(),
        ])
        
        val_dataset = LabPicsDataset(val_data, transform)
        val_dataloader = DataLoader(
            val_dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            collate_fn=collate_fn, 
            num_workers=num_workers
        )
        
        print("Start evaluating...")
        start_time = time.time()
        class_iou, mean_iou = evaluate(models, val_dataloader)
        elapsed = time.time() - start_time
        
        print("\nEvaluation Results:")
        print(f"Total time: {elapsed:.2f} seconds")
        print(f"Mean IoU: {mean_iou:.4f}")
        for i, iou in enumerate(class_iou):
            print(f"Class {i} IoU: {iou:.4f}")
        
        result_file = f"./sam_endo18_rep_eval_epoch{j}.txt"
        with open(result_file, 'w') as f:
            f.write(f"Evaluation at epoch {checkpoint_epoch}\n")
            f.write(f"Mean IoU: {mean_iou:.4f}\n")
            for i, iou in enumerate(class_iou):
                f.write(f"Class {i} IoU: {iou:.4f}\n")
        
        print(f"Results saved to {result_file}")
