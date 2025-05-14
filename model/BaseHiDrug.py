from abc import ABC

import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from pyhealth.datasets import SampleBaseDataset
from pyhealth.medcode.utils import download_and_read_json
from pyhealth.models.utils import batch_to_multihot
from pyhealth.tokenizer import Tokenizer

class BaseHiDrug(ABC, L.LightningModule):
    def __init__(
        self,
        dataset,
        feature_keys,
        label_key
    ):
        super(BaseHiDrug, self).__init__()

        self.dataset = dataset
        self.feature_keys = feature_keys
        self.label_key = label_key

        # used to query the device of the model
        self._dummy_param = nn.Parameter(torch.empty(0))
        return

    @property
    def device(self):
        return self._dummy_param.device

    def get_feature_tokenizers(self, special_tokens=None):
        if special_tokens is None:
            special_tokens = ["<pad>", "<unk>"]
        feature_tokenizers = {}
        for feature_key in self.feature_keys:
            feature_tokenizers[feature_key] = Tokenizer(
                tokens=self.dataset.get_all_tokens(key=feature_key),
                special_tokens=special_tokens,
            )
        return feature_tokenizers

    @staticmethod
    def get_embedding_layers(
        feature_tokenizers,
        embedding_dim,
    ):

        embedding_layers = nn.ModuleDict()
        for key, tokenizer in feature_tokenizers.items():
            embedding_layers[key] = nn.Embedding(
                tokenizer.get_vocabulary_size(),
                embedding_dim,
                padding_idx=tokenizer.get_padding_index(),
            )
        return embedding_layers

    @staticmethod
    def padding2d(batch):

        batch_max_length = max([len(x) for x in batch])

        # get mask
        mask = torch.zeros(len(batch), batch_max_length, dtype=torch.bool)
        for i, x in enumerate(batch):
            mask[i, : len(x)] = 1

        # level-2 padding
        batch = [x + [[0.0] * len(x[0])] * (batch_max_length - len(x)) for x in batch]

        return batch, mask

    @staticmethod
    def padding3d(batch):

        batch_max_length_level2 = max([len(x) for x in batch])
        batch_max_length_level3 = max(
            [max([len(x) for x in visits]) for visits in batch]
        )

        # the most inner vector length
        vec_len = len(batch[0][0][0])

        # get mask
        mask = torch.zeros(
            len(batch),
            batch_max_length_level2,
            batch_max_length_level3,
            dtype=torch.bool,
        )
        for i, visits in enumerate(batch):
            for j, x in enumerate(visits):
                mask[i, j, : len(x)] = 1

        # level-2 padding
        batch = [
            x + [[[0.0] * vec_len]] * (batch_max_length_level2 - len(x)) for x in batch
        ]

        # level-3 padding
        batch = [
            [x + [[0.0] * vec_len] * (batch_max_length_level3 - len(x)) for x in visits]
            for visits in batch
        ]

        return batch, mask

    def add_feature_transform_layer(self, feature_key: str, info, special_tokens=None):
        if info["type"] == str:
            # feature tokenizer
            if special_tokens is None:
                special_tokens = ["<pad>", "<unk>"]
            tokenizer = Tokenizer(
                tokens=self.dataset.get_all_tokens(key=feature_key),
                special_tokens=special_tokens,
            )
            self.feat_tokenizers[feature_key] = tokenizer
            # feature embedding
            if self.pretrained_emb != None:
                print(f"Loading pretrained embedding for {feature_key}...")
                # load pretrained embedding
                (
                    feature_embedding_dict,
                    special_tokens_embedding_dict,
                ) = self.get_pretrained_embedding(
                    feature_key, special_tokens, self.pretrained_emb
                )
                emb = []
                for i in range(tokenizer.get_vocabulary_size()):
                    idx2token = tokenizer.vocabulary.idx2token
                    if idx2token[i] in special_tokens:
                        emb.append(special_tokens_embedding_dict[idx2token[i]])
                    else:
                        emb.append(feature_embedding_dict[idx2token[i]])
                emb = torch.FloatTensor(emb)
                pretrained_emb_dim = emb.shape[1]

                self.embeddings[feature_key] = nn.Embedding.from_pretrained(
                    emb,
                    padding_idx=tokenizer.get_padding_index(),
                    freeze=False,
                )

                self.linear_layers[feature_key] = nn.Linear(
                    pretrained_emb_dim, self.embedding_dim
                )


            else:
                self.embeddings[feature_key] = nn.Embedding(
                    tokenizer.get_vocabulary_size(),
                    self.embedding_dim,
                    padding_idx=tokenizer.get_padding_index(),
                )
        elif info["type"] in [float, int]:
            self.linear_layers[feature_key] = nn.Linear(info["len"], self.embedding_dim)
        else:
            raise ValueError("Unsupported feature type: {}".format(info["type"]))

    def get_pretrained_embedding(
        self, feature_key: str, special_tokens=None, pretrained_type="LM/clinicalbert"
    ):
        feature_embedding_file = f"embeddings/{pretrained_type}/{feature_key}/{self.dataset.code_vocs[feature_key].lower()}.json"
        feature_embedding = download_and_read_json(feature_embedding_file)

        if special_tokens is not None:
            special_tokens_embedding_file = (
                f"embeddings/{pretrained_type}/special_tokens/special_tokens.json"
            )
            special_tokens_embedding = download_and_read_json(
                special_tokens_embedding_file
            )
        else:
            special_tokens_embedding = None

        return feature_embedding, special_tokens_embedding

    def get_label_tokenizer(self, special_tokens=None):
        if special_tokens is None:
            special_tokens = []
        label_tokenizer = Tokenizer(
            self.dataset.get_all_tokens(key=self.label_key),
            special_tokens=special_tokens,
        )
        return label_tokenizer

    def get_output_size(self, label_tokenizer) :

        output_size = label_tokenizer.get_vocabulary_size()
        if self.mode == "binary":
            assert output_size == 2
            output_size = 1
        return output_size

    def get_loss_function(self):

        if self.mode == "binary":
            return F.binary_cross_entropy_with_logits
        elif self.mode == "multiclass":
            return F.cross_entropy
        elif self.mode == "multilabel":
            return F.binary_cross_entropy_with_logits
        else:
            raise ValueError("Invalid mode: {}".format(self.mode))

    def prepare_labels(
        self,
        labels,
        label_tokenizer):
        if self.mode in ["binary"]:
            labels = label_tokenizer.convert_tokens_to_indices(labels)
            labels = torch.FloatTensor(labels).unsqueeze(-1)
        elif self.mode in ["multiclass"]:
            labels = label_tokenizer.convert_tokens_to_indices(labels)
            labels = torch.LongTensor(labels)
        elif self.mode in ["multilabel"]:
            # convert to indices
            labels_index = label_tokenizer.batch_encode_2d(
                labels, padding=False, truncation=False
            )
            # convert to multihot
            num_labels = label_tokenizer.get_vocabulary_size()
            labels = batch_to_multihot(labels_index, num_labels)
        else:
            raise NotImplementedError
        labels = labels.to(self.device)
        return labels

    def prepare_y_prob(self, logits):
        if self.mode in ["binary"]:
            y_prob = torch.sigmoid(logits)
        elif self.mode in ["multiclass"]:
            y_prob = F.softmax(logits, dim=-1)
        elif self.mode in ["multilabel"]:
            y_prob = torch.sigmoid(logits)
        else:
            raise NotImplementedError
        return y_prob