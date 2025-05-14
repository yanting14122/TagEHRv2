from collections import defaultdict, deque
import re
import logging
import os
import torch
import numpy as np
from sklearn.metrics import average_precision_score
from torch.nn.utils.rnn import pad_sequence

logger = logging.getLogger(__name__)

icd9_cond_lookup = {
    **{f"{i:03d}": 0 for i in range(1, 140)},    # 001–139
    **{f"{i:03d}": 1 for i in range(140, 240)},  # 140–239
    **{f"{i:03d}": 2 for i in range(240, 280)},  # 240–279
    **{f"{i:03d}": 3 for i in range(280, 290)},  # 280–289
    **{f"{i:03d}": 4 for i in range(290, 320)},  # 290–319
    **{f"{i:03d}": 5 for i in range(320, 390)},  # 320–389
    **{f"{i:03d}": 6 for i in range(390, 460)},  # 390–459
    **{f"{i:03d}": 7 for i in range(460, 520)},  # 460–519
    **{f"{i:03d}": 8 for i in range(520, 580)},  # 520–579
    **{f"{i:03d}": 9 for i in range(580, 630)},  # 580–629
    **{f"{i:03d}": 10 for i in range(630, 680)}, # 630–679
    **{f"{i:03d}": 11 for i in range(680, 710)}, # 680–709
    **{f"{i:03d}": 12 for i in range(710, 740)}, # 710–739
    **{f"{i:03d}": 13 for i in range(740, 760)}, # 740–759
    **{f"{i:03d}": 14 for i in range(760, 780)}, # 760–779
    **{f"{i:03d}": 15 for i in range(780, 800)}, # 780–799
    **{f"{i:03d}": 16 for i in range(800, 1000)},# 800–999
    **{f"V{i:02d}": 17 for i in range(1, 92)},    # V01–V91
    **{f"E{i:03d}": 18 for i in range(0, 1000)}   # E000–E999
}

icd9_proc_lookup = {
    **{f"{i:02d}": 0 for i in range(1, 6)},      # 01–05 Operations On The Nervous System
    **{f"{i:02d}": 1 for i in range(6, 8)},      # 06–07 Operations On The Endocrine System
    **{f"{i:02d}": 2 for i in range(8, 17)},     # 08–16 Operations On The Eye
    **{f"{i:02d}": 3 for i in range(17, 18)},    # 17–17 Other Miscellaneous Diagnostic And Therapeutic Procedures
    **{f"{i:02d}": 4 for i in range(18, 21)},    # 18–20 Operations On The Ear
    **{f"{i:02d}": 5 for i in range(21, 30)},    # 21–29 Operations On The Nose, Mouth, And Pharynx
    **{f"{i:02d}": 6 for i in range(30, 35)},    # 30–34 Operations On The Respiratory System
    **{f"{i:02d}": 7 for i in range(35, 40)},    # 35–39 Operations On The Cardiovascular System
    **{f"{i:02d}": 8 for i in range(40, 42)},    # 40–41 Operations On The Hemic And Lymphatic System
    **{f"{i:02d}": 9 for i in range(42, 55)},    # 42–54 Operations On The Digestive System
    **{f"{i:02d}": 10 for i in range(55, 60)},   # 55–59 Operations On The Urinary System
    **{f"{i:02d}": 11 for i in range(60, 65)},   # 60–64 Operations On The Male Genital Organs
    **{f"{i:02d}": 12 for i in range(65, 72)},   # 65–71 Operations On The Female Genital Organs
    **{f"{i:02d}": 13 for i in range(72, 76)},   # 72–75 Obstetrical Procedures
    **{f"{i:02d}": 14 for i in range(76, 85)},   # 76–84 Operations On The Musculoskeletal System
    **{f"{i:02d}": 15 for i in range(85, 87)},   # 85–86 Operations On The Integumentary System
    **{f"{i:02d}": 16 for i in range(87, 100)}   # 87–99 Miscellaneous Diagnostic And Therapeutic Procedures
}




class CodeNode:
    def __init__(self, name):
        self.name = name
        self.children = {}

    def insert(self, parts):
        current = self
        for part in parts:
            if part not in current.children:
                current.children[part] = CodeNode(part)
            current = current.children[part]

    def to_dict(self):
        return {child: self.children[child].to_dict() for child in self.children}


class CodeTree:
    def __init__(self, root_name, parser_fn, tokens):
        self.root = CodeNode(root_name)
        self.parser_fn = parser_fn 
        self.tokens = tokens
        self.add_all_codes_()

        self.index_dict = {}
        self.index_dict[0] = self.build_index_dict(level=1) # Function to parse code into levels
        self.index_dict[1] = self.build_index_dict(level=2)

    def add_code(self, code):
        levels = self.parser_fn(code)
        if levels:
            self.root.insert(levels)

    def add_all_codes_(self):
        for code in self.tokens:
            self.add_code(code)
        
    def return_dict(self):
        return self.root.to_dict()


    def get_children_per_level(self):
        level_counts = defaultdict(int)
        queue = deque([(self.root, 0)])

        while queue:
            node, level = queue.popleft()
            level_counts[level] += len(node.children)
            for child in node.children.values():
                queue.append((child, level + 1))

        return dict(level_counts)
    
    def get_num(self):
        max_level = 0
        leaf_nodes = []

        queue = deque([(self.root, 0)])

        while queue:
            node, level = queue.popleft()
            if not node.children:
                if level > max_level:
                    max_level = level
                    leaf_nodes = [node]
                elif level == max_level:
                    leaf_nodes.append(node)
            for child in node.children.values():
                queue.append((child, level + 1))

        return len(leaf_nodes)

    def get_group_index(self, code, level=0):
        """
        Builds the index dictionary if not already built, and gets the index of a group for a given code at a specified level.
        """
        # Get the hierarchical levels of the given code
        levels = self.parser_fn(code)
        
        # Check if the level exists and return the corresponding index from the index dictionary
        if len(levels) > level:
            return self.index_dict.get(levels[level], -1)
        return -1
    
    def get_group_index_batch(self, codes, level=0):
        """
        Takes a list of codes and returns a tensor of one-hot vectors for each code at the given level.
        Shape: (len(codes), num_groups_at_level)
        """
        num_groups = len(self.index_dict[level])
        indices = []

        for code in codes:
            levels = self.parser_fn(code)
            idx = self.index_dict[level].get(levels[level], -1)
            indices.append(idx)

        # Convert to one-hot
        one_hot = torch.zeros(len(codes), num_groups)
        for i, idx in enumerate(indices):
            if idx >= 0:
                one_hot[i, idx] = 1.0
        return one_hot

    def build_index_dict(self, level):
        """
        Builds the index dictionary for the given level (1 or 2).
        """
        index = {}
        queue = deque([(self.root, 0)])
        current_index = 0

        while queue:
            node, node_level = queue.popleft()
            if node_level == level:
                if node.name not in index:
                    index[node.name] = current_index
                    current_index += 1
            for child in node.children.values():
                queue.append((child, node_level + 1))
        return index


# ------------- Specific parsing logic below ---------------

def atc_parser(code, max_level=3):
    """
    Parses an ATC code into hierarchical levels.
    E.g., A01AB01 → ['A', 'A01', 'A01A']
    """
    code = code.strip().upper()
    if not re.match(r'^[A-Z0-9]+$', code):
        return []

    level_slices = [1, 3, 4, 5, 7]
    return [code[:i] for i in level_slices[:max_level]]


def icd9_diag_parser(code):
    """
    Parses an ICD9 code into hierarchical levels.
    E.g., 25040 → ['250', '2504', '25040']
    """
    code = code.strip().upper()
    levels = []
    

    if code.startswith('E') or code.startswith('V'):
        levels.append(code[:1])
        if len(code) >= 3:
            levels.append(code[:3])
    else:
        levels.append(icd9_cond_lookup[code[:3]])
        if len(code) >= 3:
            levels.append(code[:3])
    return levels



def icd9_proc_parser(code):
    """
    Parses an ICD9 code into hierarchical levels.
    E.g., 25040 → ['250', '2504', '25040']
    """
    code = code.strip().upper()
    levels = []
    levels.append(icd9_proc_lookup[code[:2]])

    if len(code) >= 2:
        levels.append(code[:2])
    return levels



def set_logger(log_dir, displaying=True, saving=True, debug=False):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    logger = logging.getLogger()  # get root logger

    if debug:
        level = logging.DEBUG
    else:
        level = logging.INFO

    logger.setLevel(level)

    formatter = logging.Formatter(
        "%(asctime)s-%(name)s-%(levelname)s: %(message)s", datefmt="%Y/%m/%d %H:%M:%S"
    )
    if saving:
        file_handler = logging.FileHandler(f"{log_dir}/run.log", mode="w")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    if displaying:
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)


def print_args(args, str_num=80):
    for arg, val in args.__dict__.items():
        logger.info(arg + "." * (str_num - len(arg) - len(str(val))) + str(val))


def multi_label_metric(prob, gt, ddi_adj, threshold=0.5):
    """
    prob is the output of sigmoid
    gt is a binary matrix
    """

    def jaccard(prob, gt):
        score = []
        for b in range(gt.shape[0]):
            target = np.where(gt[b] == 1)[0]
            predicted = np.where(prob[b] >= threshold)[0]
            inter = set(predicted) & set(target)
            union = set(predicted) | set(target)
            jaccard_score = 0 if union == 0 else len(inter) / len(union)
            score.append(jaccard_score)
        return np.mean(score)

    def precision_auc(pre, gt):
        all_micro = []
        for b in range(gt.shape[0]):
            all_micro.append(average_precision_score(gt[b], pre[b], average="macro"))
        return np.mean(all_micro)

    def prc_recall(prob, gt):
        score_prc = []
        score_recall = []
        for b in range(gt.shape[0]):
            target = np.where(gt[b] == 1)[0]
            predicted = np.where(prob[b] >= threshold)[0]
            inter = set(predicted) & set(target)
            prc_score = 0 if len(predicted) == 0 else len(inter) / len(predicted)
            recall_score = 0 if len(target) == 0 else len(inter) / len(target)
            score_prc.append(prc_score)
            score_recall.append(recall_score)
        return score_prc, score_recall

    def average_f1(prc, recall):
        score = []
        for idx in range(len(prc)):
            if prc[idx] + recall[idx] == 0:
                score.append(0)
            else:
                score.append(2 * prc[idx] * recall[idx] / (prc[idx] + recall[idx]))
        return np.mean(score)

    def ddi_rate_score(medications, ddi_matrix):
        all_cnt = 0
        ddi_cnt = 0
        for sample in medications:
            for i, med_i in enumerate(sample):
                for j, med_j in enumerate(sample):
                    if j <= i:
                        continue
                    all_cnt += 1
                    if ddi_matrix[med_i, med_j] == 1 or ddi_matrix[med_j, med_i] == 1:
                        ddi_cnt += 1
        if all_cnt == 0:
            return 0
        return ddi_cnt / all_cnt

    ja = jaccard(prob, gt)
    prauc = precision_auc(prob, gt)
    prc_ls, recall_ls = prc_recall(prob, gt)
    f1 = average_f1(prc_ls, recall_ls)

    pred = prob.copy()
    pred[pred >= threshold] = 1
    pred[pred < threshold] = 0
    pred_med = [np.where(item)[0] for item in pred]
    ddi = ddi_rate_score(pred_med, ddi_adj)

    return {"ja": ja, "prauc": prauc, "f1": f1, "ddi": ddi}


def pad_3D_list(data, pad_value=0.0, device='cuda:0'):
    max_len = max(len(seq) for seq in data)

    # Check tensor shape
    sample_tensor = next(t for seq in data for t in seq)  # First available tensor
    tensor_shape = sample_tensor.shape
    dtype = sample_tensor.dtype

    # Pad each sequence
    padded_data = []
    for seq in data:
        pad_count = max_len - len(seq)
        if pad_count > 0:
            pad_tensor = torch.full(tensor_shape, pad_value, dtype=dtype)
            seq = seq + [pad_tensor] * pad_count
        padded_data.append(torch.stack(seq))  # Shape: (max_len, *tensor_shape)

    # Stack all sequences into a batch
    return torch.stack(padded_data).to(device)