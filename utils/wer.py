from jiwer import wer
import os
import tensorflow as tf
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm


def compute_wer_list(gt_list, pred_list):
    assert len(gt_list) == len(pred_list), "Lists must have the same length."

    total = 0
    for gt, pred in zip(gt_list, pred_list):
        total += wer(gt, pred)

    return total / len(gt_list)


def cslr_beam_decode(logits, word_to_id, lengths, blank_index=0, beam_width=10):
    logits = np.array(logits)
    T, B, V = logits.shape

    # Shift logits so that blank is at last index (TensorFlow CTC requirement)
    if blank_index != V-1:
        # Swap blank_index with last index
        logits[..., [blank_index, V-1]] = logits[..., [V-1, blank_index]]

        # Adjust word_to_id mapping to match new blank index
        id_to_word = {v: k for k, v in word_to_id.items()}
        id_to_word[V-1] = "-"  # CTC blank
    else:
        id_to_word = {v: k for k, v in word_to_id.items()}

    logits_tf = tf.convert_to_tensor(logits, dtype=tf.float32)
    log_probs = tf.nn.log_softmax(logits_tf, axis=-1)

    decoded_sparse, _ = tf.nn.ctc_beam_search_decoder(
        inputs=log_probs,
        sequence_length=tf.convert_to_tensor(lengths, dtype=tf.int32),
        beam_width=beam_width
    )

    dense = tf.sparse.to_dense(decoded_sparse[0], default_value=-1)

    results = []
    for b in range(B):
        seq = []
        for idx in dense[b].numpy():
            if idx == -1:
                continue
            word = id_to_word.get(int(idx), "")
            if word != "-":  # skip CTC blank
                seq.append(word)
        results.append(" ".join(seq))

    return results

if __name__ == "__main__":
    # ------------------ Load gloss dictionary ------------------
    gloss_dict_path = "/Users/gufran/Developer/Projects/AI/ErrorDiffusion/assets/CSL-Daily/gloss_dict.npy"
    gloss_dict = np.load(gloss_dict_path, allow_pickle=True).item()

    # Keep only the first element if value is a list
    for k, v in gloss_dict.items():
        gloss_dict[k] = v[0]

    gloss_dict["-"] = 0  # blank token is at index 0

    # ------------------ Load logits ------------------
    logits_path = "/Users/gufran/Developer/data/sign/csldaily/logits/test"
    logits_files = sorted(os.listdir(logits_path))  # ensure consistent order
    logits_files_ids = [l.replace(".npy", "") for l in logits_files]

    logits_list = []
    lengths = []

    for f in logits_files:
        arr = np.load(os.path.join(logits_path, f), allow_pickle=True)
        tensor = torch.tensor(arr).squeeze(0)  # remove extra dim if present
        logits_list.append(tensor)
        lengths.append(tensor.shape[0])

    # Pad sequences along time dimension
    logits = pad_sequence(logits_list, batch_first=False, padding_value=0)  # [T, B, V]


    # ------------------ Load test info ------------------
    test_info_path = "/Users/gufran/Developer/Projects/AI/ErrorDiffusion/assets/CSL-Daily/test_info_ml.npy"
    test_info = np.load(test_info_path, allow_pickle=True).item()

    labels = [test_info[k]['label'] for k in test_info.keys() if test_info[k]["fileid"] in logits_files_ids]


    # ------------------ Decode in batches ------------------
    batch_size = 31
    B = logits.shape[1]

    full_decoded = []
    for i in tqdm(range(0, B, batch_size)):
        end = min(i + batch_size, B)
        logits_batch = logits[:, i:end, :].numpy()
        lengths_batch = lengths[i:end]
        decoded = cslr_beam_decode(logits_batch, gloss_dict, lengths_batch, blank_index=0, beam_width=10)
        
        full_decoded.extend(decoded)


    # ------------------ Compute WER ------------------
    print(len(labels), len(full_decoded))
    average_wer = compute_wer_list(labels, full_decoded)
    print(f"Average WER: {average_wer:.4f}")