import os
import yaml
import shutil
import argparse
import random
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F

import torch
import torch.nn as nn
from torch import optim

import warnings
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from dataset import setup_data
from modules.logger import Logger
from modules.wer import wer_list, beam_decode
from models.customMLM import CustomMLM

def set_rng_state(seed):
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def single_loop(model, optimizer, loader, train=True, epoch=None):
    total_loss = 0
    full_labels, full_decoded, full_cslr_stream_decoded = [], [], []

    if train: model.train()
    else: model.eval()

    tqdm_loader = tqdm(loader, ncols=100)
    for i, batch in enumerate(tqdm_loader):
        visual_fts, pose_fts, vid_lgt, pose_lengths, \
        cslr_label, cslr_label_lengths, errored_label_text, \
        label_text, accounted_label, accounted_label_text, pred_label_text, \
        modified_lengths = batch

        if args.include_pose:
            pose_fts["keypoint"] = pose_fts["keypoint"].to(args.device)
            pose_fts["mask"] = pose_fts["mask"].to(args.device)

        inp = {
            "visual_fts": visual_fts.permute(0, 2, 1).to(args.device),  # (B, T, D) -> (B, D, T)
            "pose_fts": pose_fts if args.include_pose else None,
            "vid_lgt": vid_lgt.to(args.device),
            "pose_lgt": pose_lengths.to(args.device) if args.include_pose else None,
            
            "cslr_labels": cslr_label.to(args.device),
            "cslr_label_lgt": cslr_label_lengths.to(args.device),

            "input_texts": errored_label_text if model.training else pred_label_text,
            "label_texts": label_text,

            "accounted_labels": accounted_label.to(args.device),
            "accounted_label_text": accounted_label_text,
            "modified_lengths": torch.tensor(modified_lengths, dtype=torch.long).to(args.device)
        }

        with torch.set_grad_enabled(model.training):
            outputs = model(inp)

        if train:
            loss = outputs
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        else:
            label_text, decoded, cslr_stream_decoded = outputs
            full_decoded.extend(decoded)
            full_labels.extend(label_text)
            full_cslr_stream_decoded.extend(cslr_stream_decoded)

        # set postfix description
        if train:
            tqdm_loader.set_postfix({
                "Epoch": epoch+1,
                "Loss": f"{loss.item():.4f}"
            })

    return total_loss, full_labels, full_decoded, full_cslr_stream_decoded


def main(args, main_logger, pred_logger):
    for k,v in vars(args).items():
        main_logger(f"{k}: {v}")
    main_logger("\n")

    train_loader, test_loader = setup_data(args)

    model = CustomMLM(
        args, gloss_dict=train_loader.dataset.gloss_dict,
        num_classes_cslr_head=train_loader.dataset.num_classes
    ).to(args.device)

    if args.mode == 'test':
        model.load_state_dict(torch.load(os.path.join(args.work_dir, 'best_model.pth'), map_location=args.device))

    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min',
        factor=args.gamma,
        patience=args.scheduler_patience,
    )
    
    best_wer = 1000
    patience = args.patience
    for epoch in range(args.epochs):
        total_loss, full_decoded, full_labels, full_cslr_stream_decoded = None, None, None, None
        if args.mode == 'train':
            total_loss, _, _, _ = single_loop(model, optimizer, train_loader, train=True, epoch=epoch)
            main_logger(f"[{epoch+1}/{args.epochs}], Training Loss: {total_loss/len(train_loader)}, - Learning Rate: {optimizer.param_groups[0]['lr']:.8f}")

        _, full_labels, full_decoded, full_cslr_stream_decoded = single_loop(model, optimizer, test_loader, train=False)
        wer_main = wer_list(full_labels, full_decoded)["wer"]
        wer_cslr_stream = wer_list(full_labels, full_cslr_stream_decoded)["wer"]
        main_logger(f"[{epoch+1}/{args.epochs}], Validation WER (Main): {wer_main:.4f}, Validation WER (CSLR Stream): {wer_cslr_stream:.4f}")

        if args.mode == "train": pred_logger(f"=== Epoch {epoch+1} Predictions ===")
        for i in range(5 if args.mode == 'train' else len(full_labels)):
            pred_logger(f"GT   : {full_labels[i]}")
            pred_logger(f"Pred : {full_decoded[i]}")
            pred_logger(f"CSLR : {full_cslr_stream_decoded[i]}\n")

        if args.mode == 'train' and wer_main < best_wer:
            best_wer = wer_main
            patience = args.patience
            torch.save(model.state_dict(), os.path.join(args.work_dir, 'best_model.pth'))
            main_logger(f"Best model saved at epoch {epoch+1} with WER: {best_wer:.4f}")

        elif args.mode == 'train':
            patience -= 1
            if patience == 0:
                main_logger("Early stopping triggered.\n")
                break
        
        scheduler.step(best_wer)
        if args.mode == 'train': main_logger(f"Patience: {patience}\n")
        pred_logger("\n")


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',dest='config', default='./configs/baseline.yaml')
    parser.add_argument('--dataset',dest='dataset', default='phoenix2014-T')
    parser.add_argument('--work-dir', dest='work_dir', default='./work_dir/test')
    parser.add_argument('--mode',dest='mode', default='train')
    args=parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    args.__dict__.update(config)
    with open(f"./configs/{args.dataset}.yaml", 'r') as f:
        config = yaml.safe_load(f)
    args.__dict__.update(config)

    if not os.path.exists("work_dir"): os.makedirs("work_dir")
    if args.mode == 'test': args.epochs = 1

    if args.mode == 'train':
        os.makedirs(args.work_dir, exist_ok=True)

    if args.mode == 'train':
        shutil.copy2(args.config, args.work_dir)
        shutil.copy2("./main.py", args.work_dir)
        shutil.copy2("./dataset.py", args.work_dir)
        shutil.copy2("./models/customMLM.py", args.work_dir)

    set_rng_state(42)
    main(args, Logger(os.path.join(args.work_dir, f'{args.mode}.log')), Logger(os.path.join(args.work_dir, f'pred.log')))