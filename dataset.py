import os
import torch
import argparse
import yaml
import numpy as np
import tensorflow as tf
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

def init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)

def setup_data(args):
    train_dataset = ResidualDataset(args, split='train')
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        drop_last=True,
        num_workers=0,
        collate_fn=train_dataset.collate_fn,
        worker_init_fn=init_fn
    )

    test_dataset = ResidualDataset(args, split="dev" if args.mode == "train" else "test")
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        drop_last=True,
        num_workers=0,
        collate_fn=test_dataset.collate_fn,
        worker_init_fn=init_fn
    )

    return train_loader, test_loader

class ResidualDataset(Dataset):
    def __init__(self, args, split='train', blank_prob=0.9):
        self.blank_prob = blank_prob
        gloss_dict_path = f"./assets/{args.dataset}/gloss_dict.npy"
        self.gloss_dict = np.load(gloss_dict_path, allow_pickle=True).item()
        self.gloss_dict = {k:v[0] for k,v in self.gloss_dict.items()}
        self.num_classes = len(self.gloss_dict) + 1

        info_dict = f"./assets/{args.dataset}/{split}_info_ml.npy"
        self.annots = np.load(info_dict,    allow_pickle=True).item()

        self.prelogits_path = f"{args.dataset_root}/sf_prelogits/{split}"
        self.logits_path = f"{args.dataset_root}/sf_logits/{split}"
        valid_instances = os.listdir(self.prelogits_path)
        self.video_names = [self.annots[i]["fileid"]+".npy" for i in range(len(self.annots)) if self.annots[i]["fileid"]+".npy" in valid_instances]
        self.labels = [self.annots[i]["label"] for i in range(len(self.annots)) if self.annots[i]["fileid"]+".npy" in valid_instances]

    def __len__(self):
        return len(self.video_names)    

    def remove_adjacent_duplicates(self, indices, values):
        new_indices = [indices[0]]
        new_values = [values[0]]

        streak = 0
        for i in range(1, len(values)):
            if values[i] == new_values[-1] and indices[i] == new_indices[-1] + 1 + streak: 
                streak += 1
                continue
            streak = 0
            new_indices.append(indices[i])
            new_values.append(values[i])

        return new_indices, new_values
    
    def compute_gt_logits(self, logits, label):
        gt_logits = logits.clone()
        
        non_zero_indices, non_zero = [], []
        for i, l in enumerate(gt_logits):
            max_index = torch.argmax(l).item()
            if max_index != 0:
                non_zero_indices.append(i)
                non_zero.append(max_index)

        non_zero_indices, non_zero = self.remove_adjacent_duplicates(non_zero_indices, non_zero)

        if len(label) < len(non_zero):
            for i in range(len(label), len(non_zero)):
                idx = non_zero_indices[i]
                gt_logits[idx, 0], gt_logits[idx, non_zero[i]] = gt_logits[idx, non_zero[i]], gt_logits[idx, 0]
            non_zero = non_zero[:len(label)]
            non_zero_indices = non_zero_indices[:len(label)]
        
        if len(label) > len(non_zero):
            missing = len(label) - len(non_zero)
            last_idx = non_zero_indices[-1]
            remaining_slots = gt_logits.shape[0] - last_idx - 1

            # Step 1: Shift existing non-zero indices if needed
            if missing > remaining_slots:
                shift_amount = missing - remaining_slots
                shifted_indices = []
                prev = -1
                for idx in non_zero_indices:
                    new_idx = max(prev + 1, idx - shift_amount)
                    shifted_indices.append(new_idx)
                    prev = new_idx
                non_zero_indices = shifted_indices

            # Step 2: Place extra labels evenly in remaining frames
            start_idx = non_zero_indices[-1] + 1
            end_idx = gt_logits.shape[0] - 1
            extra_labels = label[len(non_zero):]
            n_extra = len(extra_labels)

            if n_extra > 0:
                positions = np.linspace(start_idx, end_idx, n_extra, dtype=int)
                for pos, lbl in zip(positions, extra_labels):
                    gt_logits[pos, 0], gt_logits[pos, lbl] = gt_logits[pos, lbl], gt_logits[pos, 0]

                    # Update non_zero and non_zero_indices
                    non_zero.append(gt_logits[pos].argmax().item())
                    non_zero_indices.append(pos)


        if len(label) == len(non_zero):
            for i in range(len(non_zero)):
                idx = non_zero_indices[i]
                correct_label = label[i]
                incorrect_label = non_zero[i]

                tmp = gt_logits[idx, correct_label].clone()
                gt_logits[idx, correct_label] = gt_logits[idx, incorrect_label]
                gt_logits[idx, incorrect_label] = tmp - (0.1 if incorrect_label != correct_label else 0.0)

        return gt_logits
    
    def collate_fn(self, batch):
        visual_fts, logits_list, gt_logits_list, label, label_text = zip(*batch)

        lengths = [ft.shape[0] for ft in visual_fts]
        visual_fts = pad_sequence(visual_fts, batch_first=True, padding_value=0.0)
        logits = pad_sequence(logits_list, batch_first=True, padding_value=0.0)
        gt_logits = pad_sequence(gt_logits_list, batch_first=True, padding_value=0.0)
        lengths = torch.tensor(lengths, dtype=torch.long)
        label_lengths = torch.tensor([len(l) for l in label], dtype=torch.long)
        label_final = []
        for l in label:
            label_final.extend(l)
        label = torch.tensor(label_final, dtype=torch.long)
    
        return visual_fts, lengths, logits, gt_logits, label, label_text, label_lengths
    
    def __getitem__(self, idx):
        label_text = self.labels[idx]
        label = [self.gloss_dict[gloss] for gloss in label_text.split()]

        visual_ft = torch.tensor(np.load(os.path.join(self.prelogits_path, self.video_names[idx])).squeeze(0))
        logits = torch.tensor(np.load(os.path.join(self.logits_path, self.video_names[idx])).squeeze(0))
        gt_logits = self.compute_gt_logits(logits, label)

        return visual_ft, logits, gt_logits, label, label_text
        


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',dest='config', default='./configs/baseline.yaml')
    args=parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    args.__dict__.update(config)
    with open(f"./configs/{args.dataset}.yaml", 'r') as f:
        config = yaml.safe_load(f)
    args.__dict__.update(config)

    dataset = ResidualDataset(args, split='train')
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        drop_last=True,
        num_workers=0,
        collate_fn=dataset.collate_fn,
        shuffle=False
    )


    for idx, batch in enumerate(loader):
        visual_fts, logits, gt_logits, label, lengths = batch
        visual_fts, logits, gt_logits, label, lengths = batch
        print(visual_fts.shape)
        print(logits.shape)
        print(gt_logits.shape)
        print(label)
        print(lengths)

        print()

        print(torch.argmax(logits, dim=-1))
        print(torch.argmax(gt_logits, dim=-1))
        print(label)
        break
        