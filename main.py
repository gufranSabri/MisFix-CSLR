import os
import yaml
import shutil
from datetime import datetime
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
from utils.logger import Logger
from utils.wer import compute_wer_list, cslr_beam_decode
from model import MultiHeadRefiner

def set_rng_state(seed):
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def single_loop(model, optimizer, loader, train=True):
    total_loss = 0
    full_labels, full_decoded = [], []
    ctc, mse = nn.CTCLoss(reduction='none', zero_infinity=False), nn.MSELoss()

    if train: model.train()
    else: model.eval()
    for batch in tqdm(loader, ncols=100):
        visual_fts, lengths, logits, gt_logits, label, label_text, label_lengths = batch
        visual_fts = visual_fts.to(args.device)
        logits = logits.to(args.device)
        gt_logits = gt_logits.to(args.device)
        label = label.to(args.device)
        lengths = lengths.to(args.device)
        label_lengths = label_lengths.to(args.device)
        
        with torch.set_grad_enabled(model.training):
            visual_fts = torch.cat([visual_fts, logits], dim=-1)
            residual_logits = model(visual_fts)
        pred_logits = logits.permute(1, 0, 2) + residual_logits.permute(1, 0, 2)
        ground_residual = gt_logits - logits

        if train:
            ctc_loss = ctc(
                pred_logits.cpu().log_softmax(dim=-1),
                label.cpu().int(),
                lengths.cpu().int(),
                label_lengths.cpu().int()
            ).mean()
            gt_probs = F.softmax(gt_logits, dim=-1)
            pred_probs = F.softmax(pred_logits, dim=-1)
            kl_loss = F.kl_div(pred_probs.log(), gt_probs.permute(1,0,2), reduction='batchmean')

            mse_loss = mse(residual_logits, ground_residual)
        
            loss = args.ctc_weight * ctc_loss + args.mse_weight * mse_loss + args.kl_weight * kl_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        else:
            decoded = cslr_beam_decode(
                pred_logits.detach().cpu().numpy(), 
                loader.dataset.gloss_dict, 
                lengths.cpu(), 
                blank_index=0, 
                beam_width=10
            )
            full_decoded.extend(decoded)
            full_labels.extend(label_text)

    return total_loss, full_labels, full_decoded


def main(args, main_logger):
    for k,v in vars(args).items():
        main_logger(f"{k}: {v}")
    main_logger("\n")

    train_loader, test_loader = setup_data(args)
    num_classes = len(train_loader.dataset.gloss_dict) + 1

    # model = nn.Linear(args.input_dim + num_classes, num_classes).to(args.device)
    model = MultiHeadRefiner(args.input_dim + num_classes, num_classes).to(args.device)
    if args.mode == 'test':
        model.load_state_dict(torch.load(os.path.join(args.work_dir, 'best_model.pth'), map_location=args.device))

    optimizer = optim.AdamW(model.parameters(), lr=0.0001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min',
        factor=args.gamma,
        patience=args.scheduler_patience,
    )
    
    best_wer = 1.0
    patience = args.patience
    for epoch in range(args.epochs):
        total_loss, full_decoded, full_labels = None, None, None
        if args.mode == 'train':
            total_loss, _,_ = single_loop(model, optimizer, train_loader, train=True)
            main_logger(f"[{epoch+1}/{args.epochs}], Training Loss: {total_loss/len(train_loader)}, - Learning Rate: {optimizer.param_groups[0]['lr']:.8f}")


        _, full_labels, full_decoded = single_loop(model, optimizer, test_loader, train=False)
        wer = compute_wer_list(full_labels, full_decoded)
        main_logger(f"[{epoch+1}/{args.epochs}], Validation WER: {wer:.4f}")

        if args.mode == 'train' and wer < best_wer:
            best_wer = wer
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


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',dest='config', default='./configs/baseline.yaml')
    parser.add_argument('--dataset',dest='dataset', default='CSL-Daily')
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

    os.makedirs(args.work_dir, exist_ok=True)

    shutil.copy2(args.config, args.work_dir)
    shutil.copy2("./main.py", args.work_dir)
    shutil.copy2("./dataset.py", args.work_dir)
    shutil.copy2("./utils/wer.py", args.work_dir)

    set_rng_state(42)
    main(args, Logger(os.path.join(args.work_dir, f'{args.mode}.log')))