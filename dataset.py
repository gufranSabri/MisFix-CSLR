import os
import yaml
import torch
import pickle
import argparse
import numpy as np
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

def init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)

def setup_data(args):
    train_dataset = ESD(args, split='train')
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        drop_last=True,
        num_workers=0,
        collate_fn=train_dataset.collate_fn,
        worker_init_fn=init_fn
    )

    test_dataset = ESD(args, split="dev" if args.mode == "train" else "test")
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        drop_last=True,
        num_workers=0,
        collate_fn=test_dataset.collate_fn,
        worker_init_fn=init_fn
    )

    return train_loader, test_loader

class ESD(Dataset):
    def __init__(self, args, split='train'):
        self.split = split
        self.blank_id = args.blank_id
        self.dataset_root = args.dataset_root
        self.include_pose = args.include_pose

        gloss_dict_path = f"./assets/{args.dataset}/gloss_dict.npy"
        self.gloss_dict = np.load(gloss_dict_path, allow_pickle=True).item()
        self.gloss_dict = {k:v[0] for k,v in self.gloss_dict.items()}
        self.gloss_dict["-"] = self.blank_id
        self.gloss_dict_inv = {v:k for k,v in self.gloss_dict.items()}

        self.num_classes = len(self.gloss_dict)
        self.p_sub = args.p_sub
        self.p_ins = args.p_ins
        self.p_del = args.p_del

        self.annots = np.load(f"./assets/{args.dataset}/{split}_info_ml.npy", allow_pickle=True).item()
        
        if "prefix" in self.annots.keys(): del self.annots["prefix"]
        gloss_key = "label" if args.dataset not in ["phoenix2014-T"] else "gloss"

        self.features_path = f"{args.dataset_root}/{args.suffix_prefix}features/{split}"
        self.logits_path = f"{args.dataset_root}/{args.suffix_prefix}logits/{split}"
        valid_instances = os.listdir(self.features_path)
        self.video_names = [self.annots[i]["fileid"]+".npy" for i in range(len(self.annots)) if self.annots[i]["fileid"]+".npy" in valid_instances]
        self.labels = [self.annots[i][gloss_key] for i in range(len(self.annots)) if self.annots[i]["fileid"]+".npy" in valid_instances]

        self._pose_init(os.path.join(args.dataset_root, f"pose.{split}"), args.dataset, split)

    def _pose_init(self, path, dataset, phase):
        self.clip_len = 400
        self.max_length = 300

        self.tmin = 0.5 if phase == 'train' else 1
        self.tmax = 1.5 if phase == 'train' else 1

        self.w = 512 if "csl" in dataset else 210
        self.h = 512 if "csl" in dataset else 260

        with open(path, "rb") as f:
            self.pose_dict = pickle.load(f)

            if dataset == "CSL-Daily":
                new_pose_dict = {}
                for key in self.pose_dict.keys():
                    new_key = f"{phase}/{key}"
                    new_pose_dict[new_key] = self.pose_dict[key]
                self.pose_dict = new_pose_dict

    def __len__(self):
        return len(self.video_names)    
        
    def simulate_substitution_error(self, label):
        new_label = []
        for gloss in label:
            if np.random.rand() < self.p_sub:
                rand_gloss = np.random.choice(list(self.gloss_dict.values()))
                new_label.append(rand_gloss)
            else:
                new_label.append(gloss)

        return new_label
    
    def simulate_deletion_error(self, label):
        new_label = []
        for gloss in label:
            if np.random.rand() < self.p_del:
                new_label.append(self.blank_id)
            else:
                new_label.append(gloss)

        return new_label
    
    def simulate_insertion_error(self, errored_label, label):
        errored_label_new = []
        new_label = []

        for i in range(len(label)):
            errored_label_new.append(errored_label[i])
            new_label.append(label[i])

            if np.random.rand() < self.p_ins:
                rand_gloss = np.random.choice(list(self.gloss_dict.values()))
                errored_label_new.append(rand_gloss)
                new_label.append(self.blank_id)

        return errored_label_new, new_label
        
    def greedy_decode(self, logits):
        if isinstance(logits, torch.Tensor):
            logits = logits.tolist()

        decoded, prev = [], None
        for token in logits:
            if token == self.blank_id:
                prev = token
                continue

            if token != prev: decoded.append(token)
            prev = token

        return decoded
    
    def space_out(self, label):
        spaced_label = []
        for gloss in label:
            spaced_label.append(gloss)
            spaced_label.append(self.blank_id)
        return spaced_label[:-1]
    
    def make_accounted_label(self, spaced_out_label):
        for i in range(1, len(spaced_out_label)-1, 2):
            choice = np.random.choice(
                [0, 1, 2],
                p=[0.34, 0.33, 0.33]
            )
            if choice == 1:
                spaced_out_label[i] = spaced_out_label[i-1]
            elif choice == 2:
                spaced_out_label[i] = spaced_out_label[i+1]

        return spaced_out_label
        

    def __getitem__(self, idx):
        visual_ft = torch.tensor(np.load(os.path.join(self.features_path, self.video_names[idx])).squeeze(0))
        pose_ft = self.pose_dict[f"{self.split}/{self.video_names[idx].replace('.npy', '')}"]["keypoint"].permute(2, 0, 1).to(torch.float32) if self.include_pose else None

        label_text = self.labels[idx]
        label = [self.gloss_dict[gloss] for gloss in label_text.split()]
        
        spaced_out_label = self.space_out(label)
        errored_label = self.simulate_substitution_error(spaced_out_label)
        errored_label = self.simulate_deletion_error(errored_label)
        accounted_label = self.make_accounted_label(spaced_out_label)

        accounted_label_text = " ".join([self.gloss_dict_inv[i] for i in accounted_label])
        errored_label_text = " ".join([self.gloss_dict_inv[i] for i in errored_label])
        errored_label_text = "-" if len(errored_label_text.strip()) == 0 else errored_label_text
        
        logits = torch.tensor(np.load(os.path.join(self.logits_path, self.video_names[idx])).squeeze(0))
        logits = torch.argmax(logits, dim=-1).numpy().tolist()
        pred_label = self.greedy_decode(logits)
        pred_label = self.space_out(pred_label)
        pred_label = [self.gloss_dict_inv[i] for i in pred_label if i in self.gloss_dict_inv]
        pred_label_text = " ".join(pred_label)

        return visual_ft, pose_ft, label, label_text, accounted_label, errored_label_text, accounted_label_text, pred_label_text
    
    def rotate_points(self, points, angle):
        center = [0, 0]
        points_centered = points - center
        rotation_matrix = np.array([[np.cos(angle), np.sin(angle)],
                                    [-np.sin(angle), np.cos(angle)]])

        points_rotated = np.dot(points_centered, rotation_matrix.T)
        points_transformed = points_rotated + center

        return points_transformed

    def augment_preprocess_inputs(self, is_train, keypoints=None):
        if is_train == 'train':
            keypoints[:, 0, :, :] /= self.w
            keypoints[:, 1, :, :] = self.h - keypoints[:, 1, :, :]
            keypoints[:, 1, :, :] /= self.h
            keypoints[:, :2, :, :] = (keypoints[:, :2, :, :] - 0.5) / 0.5
            keypoints[:, :2, :, :] = self.random_move(
                keypoints[:, :2, :, :].permute(0, 2, 3, 1).numpy()).permute(0, 3, 1, 2)
        else:
            keypoints[:, 0, :, :] /= self.w
            keypoints[:, 1, :, :] = self.h - keypoints[:, 1, :, :]
            keypoints[:, 1, :, :] /= self.h
            keypoints[:, :2, :, :] = (keypoints[:, :2, :, :] - 0.5) / 0.5
        return keypoints

    def get_selected_index(self, vlen):
        frame_index = np.arange(vlen)
        valid_len = vlen
        return frame_index, valid_len

    def random_move(self, data_numpy):
        degrees = np.random.uniform(-15, 15)
        theta = np.radians(degrees)
        p = np.random.uniform(0, 1)
        if p >= 0.5:
            data_numpy = self.rotate_points(data_numpy, theta)
        return torch.from_numpy(data_numpy)

    def pose_collate(self, batch):
        keypoint_batch, src_length_batch = [], []
        for keypoint_sample, length in batch:
            index, valid_len = self.get_selected_index(length)
            if keypoint_sample is not None:
                keypoint_batch.append(torch.stack([keypoint_sample[:, i, :] for i in index], dim=1))
            src_length_batch.append(valid_len)

        max_length = max(src_length_batch)
        padded_sgn_keypoints = []
        for keypoints, len_ in zip(keypoint_batch, src_length_batch):
            if len_ < max_length:
                padding = keypoints[:, -1, :].unsqueeze(1)
                padding = torch.tile(padding, [1, max_length - len_, 1])
                padded_keypoint = torch.cat([keypoints, padding], dim=1)
                padded_sgn_keypoints.append(padded_keypoint)
            else:
                padded_sgn_keypoints.append(keypoints)

        keypoints = torch.stack(padded_sgn_keypoints, dim=0)
        keypoints = self.augment_preprocess_inputs(self.split, keypoints)
        src_length_batch = torch.tensor(src_length_batch)
        new_src_lengths = (((src_length_batch - 1) / 2) + 1).long()
        new_src_lengths = (((new_src_lengths - 1) / 2) + 1).long()
        max_len = max(new_src_lengths)
        mask = torch.zeros(new_src_lengths.shape[0], 1, max_len)
        for i in range(new_src_lengths.shape[0]):
            mask[i, :, :new_src_lengths[i]] = 1
        mask = mask.to(torch.bool)
        src_input = {}

        src_input['keypoint'] = keypoints
        src_input['mask'] = mask
        return src_input
    
    def collate_fn(self, batch):
        visual_fts, pose_fts, cslr_label, label_text, accounted_label, errored_label_text, accounted_label_text, pred_label_text = zip(*batch)

        pose_lengths = torch.tensor([t.shape[1] for t in pose_fts if t is not None])
        pose_prepped = self.pose_collate([(p, p.shape[1]) for p in pose_fts]) if pose_fts[0] is not None else None

        vid_lgt = [ft.shape[0] for ft in visual_fts]
        visual_fts = pad_sequence(visual_fts, batch_first=True, padding_value=0.0)
        vid_lgt = torch.tensor(vid_lgt, dtype=torch.long)

        cslr_label_final, accounted_label_final = [], []
        for l in cslr_label:
            cslr_label_final.extend(l)

        cslr_label_lengths = torch.tensor([len(l) for l in cslr_label], dtype=torch.long)
        cslr_label = torch.tensor(cslr_label_final, dtype=torch.long)
        for l in accounted_label:
            accounted_label_final.extend(l)
            
        accounted_label = torch.tensor(accounted_label_final, dtype=torch.long)
        modified_lengths = [len(elt.split(" ")) for elt in accounted_label_text]

        return visual_fts, pose_prepped, vid_lgt, pose_lengths, \
            cslr_label, cslr_label_lengths, errored_label_text, \
            label_text, accounted_label, accounted_label_text, pred_label_text, \
            modified_lengths
        

if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',dest='config', default='./configs/baseline.yaml')
    parser.add_argument('--dataset',dest='dataset', default='phoenix2014-T')
    args=parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    args.__dict__.update(config)
    with open(f"./configs/{args.dataset}.yaml", 'r') as f:
        config = yaml.safe_load(f)
    args.__dict__.update(config)

    dataset = ESD(args, split='test')
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        drop_last=True,
        num_workers=0,
        collate_fn=dataset.collate_fn,
        shuffle=False
    )

    for idx, batch in enumerate(loader):
        visual_fts, pose_prepped, vid_lgt, pose_lengths, \
        cslr_label, cslr_label_lengths, errored_label_text, \
        label_text, accounted_label, accounted_label_text, pred_label_text, \
        modified_lengths = batch
        print(visual_fts.shape)
        print(pose_prepped["keypoint"].shape)
        print(vid_lgt)
        print(pose_lengths)
        print()
        print(cslr_label)
        print(cslr_label_lengths)
        print()
        print("label", label_text, len(label_text[0].split(" ")))
        print("errored_label", errored_label_text, len(errored_label_text[0].split(" ")))
        print("accounted_label_text", accounted_label_text, len(accounted_label_text[0].split(" ")))
        print("pred_label_text", pred_label_text, len(pred_label_text[0].split(" ")))
        print()
        print(modified_lengths)

        print('===========')

        if idx == 10: break