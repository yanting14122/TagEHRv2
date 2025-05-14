import os
os.chdir('/home/yan/workspace/HiDrug')
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
# from EHRmodel import GAMENet
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from pyhealth.models.utils import batch_to_multihot, get_last_visit
from pyhealth.medcode import ATC
import pandas as pd
from model.utils import CodeTree, icd9_diag_parser, icd9_proc_parser, atc_parser, multi_label_metric, pad_3D_list
from model.BaseHiDrug import BaseHiDrug
import numpy as np
import logging

logger = logging.getLogger(__name__)




class HierarchyEncoder(nn.Module):
    def __init__(self, args, code_tree, hidden_dim, out_dim):
        super().__init__()
        self.code_tree = code_tree
        self.hidden_dim = hidden_dim
        self.level_size = self.get_level_size()
        self.device = torch.device(args.dev)

        self.level_mlp = nn.ModuleList([
            nn.Linear(self.level_size[num_level], hidden_dim)
            for num_level in range(2)
        ])

        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, out_dim),
        )

    def forward(self, code_list):
        #CODE LISt: list
        level_features = []
        for lvl in range(2):
            feature_matrix = self.extract_hierarchy_feature(code_list, lvl)
            feature_matrix = feature_matrix.to(dtype=torch.float32, device=self.device)
            level_features.append(self.level_mlp[lvl](feature_matrix))  # shape [B, hidden_dim]
        out = torch.cat(level_features, dim=-1)  # [B, 2 * hidden_dim]
        out = self.mlp(out)  # [B, out_dim]
        return out.cpu()


    def get_level_size(self):
        return self.code_tree.get_children_per_level()

    def extract_hierarchy_feature(self, codes, lvl):
        return self.code_tree.get_group_index_batch(codes, level=lvl)



class TextEmbedder(nn.Module):
    def __init__(self):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
        self.model = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
        self.text_emb_size = self.model.config.hidden_size

    def forward(self, input_text, mode = 'mean'):
        input = self.tokenizer(input_text, return_tensors='pt', padding = True, truncation=True)
        input = {k: v.to(self.model.device) for k, v in input.items()}
        outputs = self.model(**input)
        assert mode in ['mean', 'cls'], 'invalid mode of text embedding'
        if mode == 'mean':
            text_embedding = outputs.last_hidden_state.mean(dim=1) #768 dimension
        else:
            text_embedding = outputs.last_hidden_state[:, 0, :] #768 dimension
        return text_embedding






class CrossModalityModule(nn.Module):
    def __init__(self, emb_hidden_size, text_hidden_size, graph_hidden_size, crossmodal_size):
        super().__init__()
        self.text_aligner = nn.Sequential(
            nn.ReLU(),
            nn.Linear(text_hidden_size, crossmodal_size)
        )
        self.graph_aligner = nn.Sequential(
            nn.ReLU(),
            nn.Linear(graph_hidden_size, crossmodal_size)
        )

        self.fuser = nn.Sequential(
            nn.Linear(emb_hidden_size + crossmodal_size*2, crossmodal_size)
        )

    def forward(self, emb, text_emb, graph_emb):

        text_emb = self.text_aligner(text_emb)
        
        graph_emb = self.graph_aligner(graph_emb)
        output = self.fuser(torch.cat((emb, text_emb, graph_emb), dim = -1))
        return output


    def get_text_embedding(self, lookup_dict, keys):
        values = pd.Series(keys).map(lookup_dict).tolist()
        return values


# class EHR_GRUEncoder(nn.Module):
#     def __init__(self, input_size, hidden_size, bidirectional=False):
#         super().__init__()
#         self.hidden_size = hidden_size
#         self.bidirectional = bidirectional
#         self.gru = nn.GRU(input_size, hidden_size, batch_first=True, bidirectional=bidirectional)

#     def forward(self, sequences):
#         lengths = torch.tensor([len(seq) for seq in sequences])
#         padded_seqs = pad_sequence(sequences, batch_first=True)

#         # Sort by descending length
#         lengths, perm_idx = lengths.sort(0, descending=True)
#         padded_seqs = padded_seqs[perm_idx]

#         packed_input = pack_padded_sequence(padded_seqs, lengths.cpu(), batch_first=True)
#         packed_output, h_n = self.gru(packed_input)

#         output, _ = pad_packed_sequence(packed_output, batch_first=True)

#         # Restore original order
#         _, unperm_idx = perm_idx.sort(0)
#         output = output[unperm_idx]
#         h_n = h_n[:, unperm_idx]

#         return output, h_n

class Predictor(nn.Module):
    def __init__(self, hidden_size, output_size) -> None:
        super().__init__()
        self.predictor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, h_n):
        logits = self.predictor(h_n).squeeze(-1)
        return logits


class PatientRepr(BaseHiDrug):
    def __init__(self, args, dataset, hidden_size, crossmodal_size, num_rnn_layers, dropout):
        super().__init__(dataset=dataset, feature_keys=["conditions", "procedures", "drugs_hist"], label_key="drugs")
        self.args = args
        self.hidden_size = hidden_size
        self.num_rnn_layers = num_rnn_layers
        self.dropout_fn = torch.nn.Dropout(dropout)
        self.feat_tokenizers = self.get_feature_tokenizers()
        self.label_tokenizer = self.get_label_tokenizer()
        self.embeddings = self.get_embedding_layers(
            self.feat_tokenizers, self.hidden_size
        )
        self.diag_size = self.get_vocab_size(self.feature_keys[0])
        self.proc_size = self.get_vocab_size(self.feature_keys[1])
        self.drug_size = self.get_vocab_size(self.label_key)
        self.code_trees = self.build_codetree()
        self.label_size = self.drug_size
        self.ddi_adj = self.generate_ddi_adj()

        
        self.hierarchy_embedder = nn.ModuleDict({
            x: HierarchyEncoder(self.args, code_tree, hidden_size*2, hidden_size)
            for code_tree, x in zip(self.code_trees, self.feature_keys)
            })

        self.cond_embedder = self.get_pretrained_embedding(self.feature_keys[0])
        self.proc_embedder = self.get_pretrained_embedding(self.feature_keys[1])
        self.proc_embedder[0].update({'239':[0.0]*768})
        self.proc_embedder[0].update({'118':[0.0]*768})
        self.proc_embedder[0].update({'309':[0.0]*768})
        self.drug_embedder = self.get_pretrained_embedding(self.label_key)
        # self.text_aligner = nn.Linear(768 , hidden_size)

        ### Baek ###
        cond_embed_dict = {self.feat_tokenizers['conditions'].convert_tokens_to_indices([k])[0]: torch.tensor(v) for k, v in self.cond_embedder[0].items()}
        cond_vocab_size = max(cond_embed_dict) + 1
        cond_embedding_dim = next(iter(cond_embed_dict.values())).shape[0]
        self.cond_embedding_table = torch.stack([cond_embed_dict.get(i, torch.zeros(cond_embedding_dim)) for i in range(cond_vocab_size)])

        proc_embed_dict = {self.feat_tokenizers['procedures'].convert_tokens_to_indices([k])[0]: torch.tensor(v) for k, v in self.proc_embedder[0].items()}
        proc_vocab_size = max(proc_embed_dict) + 1
        proc_embedding_dim = next(iter(proc_embed_dict.values())).shape[0]
        self.proc_embedding_table = torch.stack([proc_embed_dict.get(i, torch.zeros(proc_embedding_dim)) for i in range(proc_vocab_size)])

        drugs_hist_embed_dict = {self.feat_tokenizers['drugs_hist'].convert_tokens_to_indices([k])[0]: torch.tensor(v) for k, v in self.drug_embedder[0].items()}
        drugs_hist_vocab_size = max(drugs_hist_embed_dict) + 1
        drugs_hist_embedding_dim = next(iter(drugs_hist_embed_dict.values())).shape[0]
        self.drugs_hist_embedding_table = torch.stack([drugs_hist_embed_dict.get(i, torch.zeros(drugs_hist_embedding_dim)) for i in range(drugs_hist_vocab_size)])
        ### Baek ###

        
        self.crossmodality_aligner = nn.ModuleList([
            CrossModalityModule(
            hidden_size,768, hidden_size, crossmodal_size) for _ in range(3)
        ])

        self.rnns = torch.nn.ModuleDict(
            {
                x: torch.nn.GRU(
                    crossmodal_size,
                    crossmodal_size,
                    num_layers=num_rnn_layers,
                    dropout=dropout if num_rnn_layers > 1 else 0,
                    batch_first=True,
                )
                for x in self.feature_keys
            }
        )

        self.summary = nn.Linear(crossmodal_size*3, hidden_size)

    def forward(self, conditions, procedures, drugs_hist, **kwargs):
        summary, query_cond, query_proc= self.encode_patient(conditions, procedures, drugs_hist)
        return summary, query_cond, query_proc

    def get_vocab_size(self, key):
        return len(self.dataset.get_all_tokens(key))

    # def get_patient_med_hist(self, conditions, procedures, drugs_hist):
    #     condition_emb = self.encode_patient("conditions", conditions)
    #     procedure_emb = self.encode_patient("procedures", procedures)
    #     drugs_hist_emb = self.encode_patient("drugs_hist", drugs_hist)
    #     # patient_emb = torch.cat([condition_emb, procedure_emb, drugs_hist_emb], dim=-1)
    #     # med_hist = self.query_layer(patient_emb)
    #     return condition_emb, procedure_emb, drugs_hist_emb

    def get_patient_text_repr(self, conditions, procedures, drugs_hist):
        # condition_text_emb, procedure_text_emb, drugs_hist_text_emb = [], [], []
        # for patient in conditions:
        #     condition_text_emb.append([torch.tensor(self.get_text_embedding(self.cond_embedder[0], visit), dtype = torch.float32).mean(dim = 0) for visit in patient])
        # for patient in procedures:
        #     procedure_text_emb.append([torch.tensor(self.get_text_embedding(self.proc_embedder[0], visit), dtype = torch.float32).mean(dim = 0) for visit in patient])
        
        ### Baek ###
        condition_text_emb = self.text_embedding(conditions, feat_name='conditions')
        procedure_text_emb = self.text_embedding(procedures, feat_name='procedures')
        drugs_hist_text_emb = self.text_embedding(drugs_hist, feat_name='drugs_hist')
        # condition_text_emb, procedure_text_emb, drugs_hist_text_emb = [], [], []
        # for cond_patient,proc_patient in zip(conditions, procedures):
        #     condition_text_emb.append([torch.tensor(pd.Series(self.cond_embedder[0])[visit], dtype=torch.float32).mean(0) for visit in cond_patient])
        #     procedure_text_emb.append([torch.tensor(pd.Series(self.proc_embedder[0])[visit], dtype=torch.float32).mean(0) for visit in proc_patient])
        ### Baek ###

        # dim = procedure_text_emb[0][0].shape
        # for patient in drugs_hist:
        #     if len(patient[0]) >= 1:
        #         drugs_hist_text_emb.append([torch.tensor(self.get_text_embedding(self.drug_embedder[0], visit), dtype = torch.float64).mean(dim = 0) for visit in patient])
        #     else:
        #         drugs_hist_text_emb.append(torch.zeros(dim).unsqueeze(0))
        # condition_text_emb = pad_3D_list(condition_text_emb)
        # procedure_text_emb = pad_3D_list(procedure_text_emb)
        # drugs_hist_text_emb = pad_3D_list(drugs_hist_text_emb)
        # return condition_text_emb, procedure_text_emb, drugs_hist_text_emb
        return condition_text_emb.to(self.device), procedure_text_emb.to(self.device), drugs_hist_text_emb.to(self.device)

    def text_embedding(self, features, feat_name):
        if feat_name == 'drugs_hist':
            features = [feature[:-1] for feature in features]
        features_tensor = torch.tensor(self.feat_tokenizers[feat_name].batch_encode_3d(features,max_length=(128,128)), dtype=torch.long)
        mask = (features_tensor != 0).unsqueeze(-1)
        if feat_name == 'conditions':
            x = self.cond_embedding_table[features_tensor]
        elif feat_name == 'procedures':
            x = self.proc_embedding_table[features_tensor]
        elif feat_name == 'drugs_hist':
            x = self.drugs_hist_embedding_table[features_tensor]
        features_text_emb = (x * mask).sum(dim=2) / mask.sum(dim=2).clamp(min=1)

        return features_text_emb


    def get_patient_hie_repr(self, conditions, procedures, drugs_hist):
        # cond_hie_emb, proc_hie_emb, drug_hie_emb = [], [], []
        # for patient in conditions:
        #     cond_hie_emb.append([(self.hierarchy_embedder['conditions'](visit)).mean(dim = 0) for visit in patient])
                       
        # for patient in procedures:
        #     proc_hie_emb.append([(self.hierarchy_embedder['procedures'](visit)).mean(dim = 0) for visit in patient])
        # for patient in drugs_hist:
        #     if len(patient[0]) >= 1:
        #         drug_hie_emb.append([(self.hierarchy_embedder['drugs_hist'](visit)).mean(dim = -1) for visit in patient])
        #     else:
        #         drug_hie_emb.append(torch.zeros(1))

        ### Baek ###
        cond_hie_emb, proc_hie_emb, drug_hie_emb = [], [], []
        for cond_patient, proc_patient, drug_patient in zip(conditions,procedures,drugs_hist):
            cond_hie_emb.append([(self.hierarchy_embedder['conditions'](visit)).mean(dim = 0) for visit in cond_patient])
            proc_hie_emb.append([(self.hierarchy_embedder['procedures'](visit)).mean(dim = 0) for visit in proc_patient])
            if len(drug_patient[0]) >= 1:
                drug_hie_emb.append([(self.hierarchy_embedder['drugs_hist'](visit)).mean(dim = 0) for visit in drug_patient[:-1]])
            else:
                drug_hie_emb.append([])
                
        ### Baek ###

        cond_hie_emb = pad_3D_list(cond_hie_emb)
        proc_hie_emb = pad_3D_list(proc_hie_emb)
        drug_hie_emb = pad_3D_list(drug_hie_emb)


        # drug_hie_emb = pad_3D_list([drug_hie_emb])
        return cond_hie_emb, proc_hie_emb, drug_hie_emb
        # return cond_hie_emb, proc_hie_emb, drug_hie_emb

    def build_codetree(self):
        parser = {'conditions': icd9_diag_parser, 'procedures': icd9_proc_parser, 'drugs_hist': atc_parser}
        code_trees = []
        for i in range(len(self.feature_keys)):
            dict_ = self.dataset.get_all_tokens(key = self.feature_keys[i])
            code_tree = CodeTree('root', parser[self.feature_keys[i]], dict_)
            code_trees.append(code_tree)
        return code_trees

    def get_raw_embedding(self, feature_key, raw_values):
        codes = self.feat_tokenizers[feature_key].batch_encode_3d(
            raw_values, truncation=(False, False)
        )
        tensor_codes = torch.tensor(
            codes, dtype=torch.long, device=self.device
        )  # [bs, v_len, code_len]

        embeddings = self.embeddings[feature_key](
            tensor_codes
        )  # [bs, v_len, code_len, dim]
        return embeddings

    def encode_patient(self, conditions, procedures, drugs_hist):
        cond_feat_emb = (self.get_raw_embedding('conditions', conditions)).mean(dim = -2)
        proc_feat_emb = (self.get_raw_embedding('procedures', procedures)).mean(dim = -2)
        drugs_hist_feat_emb = (self.get_raw_embedding('drugs_hist', drugs_hist)).mean(dim = -2)
        # if len(drugs_hist[0]) >= 1:
        #     drug_feat_emb = (self.get_raw_embedding('drugs_hist', drugs_hist)).mean(dim = -2)
        # else:
        #     drug_feat_emb = drugs_hist
        condition_text_emb, procedure_text_emb, drugs_hist_text_emb = self.get_patient_text_repr(conditions, procedures, drugs_hist)
        cond_hie_emb, proc_hie_emb, drugs_hist_feat_emb= self.get_patient_hie_repr(conditions, procedures, drugs_hist)
        mask = torch.sum(cond_hie_emb, dim=-1) != 0
        
        conds = self.crossmodality_aligner[0](cond_feat_emb, condition_text_emb, cond_hie_emb)
        procs = self.crossmodality_aligner[1](proc_feat_emb, procedure_text_emb, proc_hie_emb)
        drugs = self.crossmodality_aligner[2](drugs_hist_feat_emb, drugs_hist_text_emb, drugs_hist_feat_emb)
        # self.crossmodality_aligner[2](drug_feat_emb, drugs_hist_text_emb, drug_hie_emb)
        
        conds = conds * mask.unsqueeze(-1)
        procs = procs * mask.unsqueeze(-1)
        drugs = drugs * mask[:, :-1].unsqueeze(-1)
        # lengths = torch.sum(mask.int(), dim=-1).cpu()
        # embeddings_packed = pack_padded_sequence(
        #     embeddings, lengths, batch_first=True, enforce_sorted=False
        # )
        lengths = mask.sum(dim=1)  # shape: [B]
        last_indices = lengths - 1 

        conds, cond_hn = self.rnns[self.feature_keys[0]](conds)
        procs, proc_hn = self.rnns[self.feature_keys[1]](procs)
        drugs, drug_hn = self.rnns[self.feature_keys[2]](drugs)

        summary = self.summary(torch.cat([cond_hn[-1], proc_hn[-1], drug_hn[-1]], dim = -1))
        query_cond = conds[torch.arange(conds.size(0)), last_indices] 
        query_proc = procs[torch.arange(procs.size(0)), last_indices] 
            # outputs, _ = pad_packed_sequence(outputs_packed, batch_first=True)
        return summary, query_cond, query_proc


    def generate_ddi_adj(self) -> torch.FloatTensor:
        """Generates the DDI graph adjacency matrix."""
        atc = ATC()
        ddi = atc.get_ddi(gamenet_ddi=True)
        vocab_to_index = self.label_tokenizer.vocabulary
        ddi_adj = np.zeros((self.label_size, self.label_size))
        ddi_atc3 = [
            [ATC.convert(l[0], level=3), ATC.convert(l[1], level=3)] for l in ddi
        ]
        for atc_i, atc_j in ddi_atc3:
            if atc_i in vocab_to_index and atc_j in vocab_to_index:
                ddi_adj[vocab_to_index(atc_i), vocab_to_index(atc_j)] = 1
                ddi_adj[vocab_to_index(atc_j), vocab_to_index(atc_i)] = 1
        ddi_adj = torch.FloatTensor(ddi_adj)
        return ddi_adj
    def get_text_embedding(self, lookup_dict, keys):
        if all([i in lookup_dict for i in keys]):
            values = pd.Series(keys).map(lookup_dict).tolist()
        else:
            values = [lookup_dict.get(i, [0] *768) for i in keys]
        return values


# class HiDrug(BaseHiDrug):
#     def __init__(self, args, dataset, graph_hidden_size, num_rnn_layers, crossmodal_size, dropout=0.2, bidirectional=False):
#         super().__init__(dataset=dataset, feature_keys=["conditions", "procedures", "drugs_hist"], label_key="drugs")
#         # self.embeddings = nn.ModuleList(
#         #      [nn.Embedding(voc_size[i], crossmodal_size) for i in range(3)])
#         self.dataset = dataset
#         self.args = args
#         self.best_monitor_metric = 0.0
#         # self.feat_tokenizers = self.get_feature_tokenizers()
#         self.label_tokenizer = self.get_label_tokenizer()
#         # self.embeddings = self.get_embedding_layers(
#         #     self.feat_tokenizers, crossmodal_size
#         # )
#         # self.code_trees = self.build_codetree()
#         # self.diag_size = self.get_vocab_size(self.feature_keys[0])
#         # self.proc_size = self.get_vocab_size(self.feature_keys[1])
#         # self.drug_size = self.get_vocab_size(self.label_key)
#         # self.label_size = self.drug_size

#         self.patient_representer = PatientRepr(args, dataset, graph_hidden_size, crossmodal_size, num_rnn_layers, dropout)


#         # self.ehrmodel = EHR_GRUEncoder(input_size=crossmodal_size, hidden_size=crossmodal_size*2, bidirectional=bidirectional)
#         # self.ehrmodel = GAMENet(vocab_size=voc_size, ehr_adj=ehr_adj, ddi_adj=ddi_adj,
#         #                         emb_dim=crossmodal_size, device=torch.device('cpu:0'), ddi_in_memory=True)
#         self.predictor = Predictor(graph_hidden_size + crossmodal_size*2, 
#                                    self.patient_representer.drug_size)



#     def forward(self, conditions, procedures, drugs_hist):
#         summary, query_cond, query_proc= self.patient_representer(conditions, procedures, drugs_hist)
#         out = self.predictor(torch.cat([summary, query_cond, query_proc], dim = -1))
#         return {"logits": out}

#     # def encode_medical_history(self, conditions, procedures, drugs_hist):
#     #     cond_feat = self.hierarchy_embedder[0](conditions)
#     #     proc_feat = self.hierarchy_embedder[1](procedures)
#     #     drugs_hist_feat = self.hierarchy_embedder[2](drugs_hist)
#     #     med_hist = torch.cat([cond_feat, proc_feat, drugs_hist_feat], dim=-1)
#     #     med_hist = self.query_layer(med_hist)
#     #     return med_hist

#     def encode_medical_history(self, conditions, procedures, drugs_hist):
#         cond_feat = self.hierarchy_embedder[0](conditions)
#         proc_feat = self.hierarchy_embedder[1](procedures)
#         drugs_hist_feat = self.hierarchy_embedder[2](drugs_hist)
#         return cond_feat, proc_feat, drugs_hist_feat

#     def get_text_embedding(self, lookup_dict, keys):
#         values = pd.Series(keys).map(lookup_dict).tolist()
#         return values


#     def encode_patient(
#         self, feature_key: str, raw_values):
#         codes = self.feat_tokenizers[feature_key].batch_encode_3d(
#             raw_values, truncation=(False, False)
#         )
#         tensor_codes = torch.tensor(
#             codes, dtype=torch.long, device=self.device
#         )  # [bs, v_len, code_len]

#         embeddings = self.embeddings[feature_key](
#             tensor_codes
#         )  # [bs, v_len, code_len, dim]
#         return embeddings


#     def training_step(self, batch, batch_idx):
#         conditions = batch["conditions"]
#         procedures = batch["procedures"]
#         drugs_hist = batch["drugs_hist"]
#         drugs = batch["drugs"]

#         bce_labels, multi_labels = self.prepare_labels(drugs)

#         output = self.forward(conditions, procedures, drugs_hist)

#         loss = self.calc_loss(output["logits"], bce_labels, multi_labels)

#         self.log("train/loss", loss)
#         return loss

#     def on_validation_epoch_start(self):
#         self.y_true_all = []
#         self.y_prob_all = []

#     def validation_step(self, batch, batch_idx):
#         conditions = batch["conditions"]
#         procedures = batch["procedures"]
#         drugs_hist = batch["drugs_hist"]
#         drugs = batch["drugs"]

#         # prepare labels
#         labels_index = self.label_tokenizer.batch_encode_2d(
#             drugs, padding=False, truncation=False
#         )
#         # convert to multihot
#         labels = batch_to_multihot(labels_index, self.patient_representer.label_size)

#         output = self.forward(conditions, procedures, drugs_hist)
#         y_prob = torch.sigmoid(output["logits"])

#         y_true = labels.cpu().numpy()
#         y_prob = y_prob.cpu().numpy()

#         self.y_true_all.append(y_true)
#         self.y_prob_all.append(y_prob)
#         return

#     def on_validation_epoch_end(self):
#         y_true_all = np.concatenate(self.y_true_all, axis=0)
#         y_prob_all = np.concatenate(self.y_prob_all, axis=0)
#         self.y_true_all.clear()
#         self.y_prob_all.clear()

#         scores = multi_label_metric(
#             y_prob_all, y_true_all, self.patient_representer.ddi_adj
#         )

#         for key in scores.keys():
#             self.log(f"val/{key}", scores[key])

#         monitor_metric = scores["ja"]
#         if monitor_metric > self.best_monitor_metric:
#             self.best_monitor_metric = monitor_metric
#             logger.info(
#                 f"New best ja: {self.best_monitor_metric:.4f} in epoch {self.trainer.current_epoch}"
#             )

#         self.log("val/best_ja", self.best_monitor_metric)

#     def on_test_epoch_start(self):
#         self.y_true_all = []
#         self.y_prob_all = []

#     def test_step(self, batch, batch_idx):
#         conditions = batch["conditions"]
#         procedures = batch["procedures"]
#         drugs_hist = batch["drugs_hist"]
#         drugs = batch["drugs"]

#         # prepare labels
#         labels_index = self.label_tokenizer.batch_encode_2d(
#             drugs, padding=False, truncation=False
#         )
#         # convert to multihot
#         labels = batch_to_multihot(labels_index, self.patient_representer.label_size)

#         output = self.forward(conditions, procedures, drugs_hist)
#         y_prob = torch.sigmoid(output["logits"])

#         y_true = labels.cpu().numpy()
#         y_prob = y_prob.cpu().numpy()

#         self.y_true_all.append(y_true)
#         self.y_prob_all.append(y_prob)
#         return

#     def on_test_epoch_end(self):
#         y_true_all = np.concatenate(self.y_true_all, axis=0)
#         y_prob_all = np.concatenate(self.y_prob_all, axis=0)
#         self.y_true_all.clear()
#         self.y_prob_all.clear()

#         scores = multi_label_metric(
#             y_prob_all, y_true_all, self.patient_representer.ddi_adj
#         )

#         for key in scores.keys():
#             logger.info(f"test/{key}: {scores[key]}")
#             self.log(f"test/{key}", scores[key])
#         return scores

#     def configure_optimizers(self):
#         opt = torch.optim.Adam(self.parameters(), lr=self.args.lr, weight_decay=1e-5)
#         return [opt]

#     def prepare_labels(self, drugs):
#         # prepare labels
#         labels_index = self.patient_representer.label_tokenizer.batch_encode_2d(
#             drugs, padding=False, truncation=False
#         )
#         # convert to multihot
#         labels = batch_to_multihot(labels_index, self.patient_representer.label_size)

#         multi_labels = -np.ones(
#             (len(labels), self.patient_representer.label_size), dtype=np.int64
#         )
#         for idx, cont in enumerate(labels_index):
#             # remove redundant labels
#             cont = list(set(cont))
#             multi_labels[idx, : len(cont)] = cont
#         multi_labels = torch.from_numpy(multi_labels)
#         return labels.to(self.device), multi_labels.to(self.device)

#     def calc_loss(self, logits, bce_labels, multi_labels):
#         loss_bce = F.binary_cross_entropy_with_logits(logits, bce_labels)
#         loss_multi = F.multilabel_margin_loss(torch.sigmoid(logits), multi_labels)
#         loss_task = 0.95 * loss_bce + 0.05 * loss_multi
#         return loss_task


class HiDrug(BaseHiDrug):
    def __init__(self, args, dataset, graph_hidden_size, num_rnn_layers, crossmodal_size, dropout=0.2, bidirectional=False):
        super().__init__(dataset=dataset, feature_keys=["conditions", "procedures", "drugs_hist"], label_key="drugs")
        # self.embeddings = nn.ModuleList(
        #      [nn.Embedding(voc_size[i], crossmodal_size) for i in range(3)])
        self.dataset = dataset
        self.args = args
        self.best_monitor_metric = 0.0
        # self.feat_tokenizers = self.get_feature_tokenizers()
        self.label_tokenizer = self.get_label_tokenizer()
        # self.embeddings = self.get_embedding_layers(
        #     self.feat_tokenizers, crossmodal_size
        # )
        # self.code_trees = self.build_codetree()
        # self.diag_size = self.get_vocab_size(self.feature_keys[0])
        # self.proc_size = self.get_vocab_size(self.feature_keys[1])
        # self.drug_size = self.get_vocab_size(self.label_key)
        # self.label_size = self.drug_size

        self.patient_representer = PatientRepr(args, dataset, graph_hidden_size, crossmodal_size, num_rnn_layers, dropout)


        # self.ehrmodel = EHR_GRUEncoder(input_size=crossmodal_size, hidden_size=crossmodal_size*2, bidirectional=bidirectional)
        # self.ehrmodel = GAMENet(vocab_size=voc_size, ehr_adj=ehr_adj, ddi_adj=ddi_adj,
        #                         emb_dim=crossmodal_size, device=torch.device('cpu:0'), ddi_in_memory=True)
        self.predictor = Predictor(graph_hidden_size + crossmodal_size*2, 
                                   self.patient_representer.drug_size)



    def forward(self, conditions, procedures, drugs_hist):
        summary, query_cond, query_proc= self.patient_representer(conditions, procedures, drugs_hist)
        out = self.predictor(torch.cat([summary, query_cond, query_proc], dim = -1))
        return {"logits": out}

    # def encode_medical_history(self, conditions, procedures, drugs_hist):
    #     cond_feat = self.hierarchy_embedder[0](conditions)
    #     proc_feat = self.hierarchy_embedder[1](procedures)
    #     drugs_hist_feat = self.hierarchy_embedder[2](drugs_hist)
    #     med_hist = torch.cat([cond_feat, proc_feat, drugs_hist_feat], dim=-1)
    #     med_hist = self.query_layer(med_hist)
    #     return med_hist

    def encode_medical_history(self, conditions, procedures, drugs_hist):
        cond_feat = self.hierarchy_embedder[0](conditions)
        proc_feat = self.hierarchy_embedder[1](procedures)
        drugs_hist_feat = self.hierarchy_embedder[2](drugs_hist)
        return cond_feat, proc_feat, drugs_hist_feat

    def get_text_embedding(self, lookup_dict, keys):
        values = pd.Series(keys).map(lookup_dict).tolist()
        return values


    def encode_patient(
        self, feature_key: str, raw_values):
        codes = self.feat_tokenizers[feature_key].batch_encode_3d(
            raw_values, truncation=(False, False)
        )
        tensor_codes = torch.tensor(
            codes, dtype=torch.long, device=self.device
        )  # [bs, v_len, code_len]

        embeddings = self.embeddings[feature_key](
            tensor_codes
        )  # [bs, v_len, code_len, dim]
        return embeddings


    def training_step(self, batch, batch_idx):
        conditions = batch["conditions"]
        procedures = batch["procedures"]
        drugs_hist = batch["drugs_hist"]
        drugs = batch["drugs"]

        bce_labels, multi_labels = self.prepare_labels(drugs)

        output = self.forward(conditions, procedures, drugs_hist)

        loss = self.calc_loss(output["logits"], bce_labels, multi_labels)

        self.log("train/loss", loss)
        return loss

    def on_validation_epoch_start(self):
        self.y_true_all = []
        self.y_prob_all = []

    def validation_step(self, batch, batch_idx):
        conditions = batch["conditions"]
        procedures = batch["procedures"]
        drugs_hist = batch["drugs_hist"]
        drugs = batch["drugs"]

        # prepare labels
        labels_index = self.label_tokenizer.batch_encode_2d(
            drugs, padding=False, truncation=False
        )
        # convert to multihot
        labels = batch_to_multihot(labels_index, self.patient_representer.label_size)

        output = self.forward(conditions, procedures, drugs_hist)
        y_prob = torch.sigmoid(output["logits"])

        y_true = labels.cpu().numpy()
        y_prob = y_prob.cpu().numpy()

        self.y_true_all.append(y_true)
        self.y_prob_all.append(y_prob)
        return

    def on_validation_epoch_end(self):
        y_true_all = np.concatenate(self.y_true_all, axis=0)
        y_prob_all = np.concatenate(self.y_prob_all, axis=0)
        self.y_true_all.clear()
        self.y_prob_all.clear()

        scores = multi_label_metric(
            y_prob_all, y_true_all, self.patient_representer.ddi_adj
        )

        for key in scores.keys():
            self.log(f"val/{key}", scores[key])

        monitor_metric = scores["ja"]
        if monitor_metric > self.best_monitor_metric:
            self.best_monitor_metric = monitor_metric
            logger.info(
                f"New best ja: {self.best_monitor_metric:.4f} in epoch {self.trainer.current_epoch}"
            )

        self.log("val/best_ja", self.best_monitor_metric)

    def on_test_epoch_start(self):
        self.y_true_all = []
        self.y_prob_all = []

    def test_step(self, batch, batch_idx):
        conditions = batch["conditions"]
        procedures = batch["procedures"]
        drugs_hist = batch["drugs_hist"]
        drugs = batch["drugs"]

        # prepare labels
        labels_index = self.label_tokenizer.batch_encode_2d(
            drugs, padding=False, truncation=False
        )
        # convert to multihot
        labels = batch_to_multihot(labels_index, self.patient_representer.label_size)

        output = self.forward(conditions, procedures, drugs_hist)
        y_prob = torch.sigmoid(output["logits"])

        y_true = labels.cpu().numpy()
        y_prob = y_prob.cpu().numpy()

        self.y_true_all.append(y_true)
        self.y_prob_all.append(y_prob)
        return

    def on_test_epoch_end(self):
        y_true_all = np.concatenate(self.y_true_all, axis=0)
        y_prob_all = np.concatenate(self.y_prob_all, axis=0)
        self.y_true_all.clear()
        self.y_prob_all.clear()

        scores = multi_label_metric(
            y_prob_all, y_true_all, self.patient_representer.ddi_adj
        )

        for key in scores.keys():
            logger.info(f"test/{key}: {scores[key]}")
            self.log(f"test/{key}", scores[key])
        return scores

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=self.args.lr, weight_decay=1e-5)
        return [opt]

    def prepare_labels(self, drugs):
        # prepare labels
        labels_index = self.patient_representer.label_tokenizer.batch_encode_2d(
            drugs, padding=False, truncation=False
        )
        # convert to multihot
        labels = batch_to_multihot(labels_index, self.patient_representer.label_size)

        multi_labels = -np.ones(
            (len(labels), self.patient_representer.label_size), dtype=np.int64
        )
        for idx, cont in enumerate(labels_index):
            # remove redundant labels
            cont = list(set(cont))
            multi_labels[idx, : len(cont)] = cont
        multi_labels = torch.from_numpy(multi_labels)
        return labels.to(self.device), multi_labels.to(self.device)

    def calc_loss(self, logits, bce_labels, multi_labels):
        loss_bce = F.binary_cross_entropy_with_logits(logits, bce_labels)
        loss_multi = F.multilabel_margin_loss(torch.sigmoid(logits), multi_labels)
        loss_task = 0.95 * loss_bce + 0.05 * loss_multi
        return loss_task