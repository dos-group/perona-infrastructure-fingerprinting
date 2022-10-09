import itertools

import pytorch_lightning as pl
import pandas as pd
import torch
import torch.nn.functional as F
import torch_geometric.nn
from pytorch_metric_learning.distances import CosineSimilarity
from torch import nn
from torch_geometric.nn import Sequential, TransformerConv
from torch_geometric.utils import dropout_adj
from pytorch_metric_learning import miners, losses
from torchvision import ops as O
import torchmetrics
import collections
import numpy as np

from config import GeneralConfig


@torch.no_grad()
def init_weights(m):
    """Weight / Bias Initialization for linear layers."""
    if isinstance(m, (torch.nn.Linear,)):
        # According to
        # 1) https://pytorch.org/docs/1.11/nn.init.html?highlight=init
        # 2) https://pytorch.org/docs/1.11/generated/torch.nn.SELU.html?highlight=selu#torch.nn.SELU
        # --> nonlinearity='linear'
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='linear')
        # biases zero
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


class PeronaGraphModel(pl.LightningModule):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.save_hyperparameters()

        self.input_dim: int = kwargs["input_dim"]
        self.hidden_dim: int = kwargs["hidden_dim"]
        self.hidden_dim_mid: int = int(self.hidden_dim * 2)
        self.output_dim: int = kwargs["output_dim"]
        self.predecessor_dim: int = kwargs["predecessor_dim"]

        # we need the following to account for class imbalance
        # --> (used in last linear layer bias init)
        self.next_pos_neg_ratio = torch.FloatTensor([kwargs["next_pos_sample_count"] / kwargs["next_neg_sample_count"]])
        # --> (pos-weighted binary cross-entropy)
        self.next_neg_pos_ratio = torch.FloatTensor([kwargs["next_neg_sample_count"] / kwargs["next_pos_sample_count"]])
        # --> (class-balanced focal loss)
        self.focal_gamma: float = kwargs['focal_gamma']
        classbalanced_beta: float = kwargs['classbalanced_beta']
        effective_num = 1.0 - np.power(classbalanced_beta,
                                       [kwargs["next_pos_sample_count"], kwargs["next_neg_sample_count"]])
        weights = (1.0 - classbalanced_beta) / np.array(effective_num)
        self.focal_alpha: float = (weights / np.sum(weights))[0]

        self.margin: float = kwargs["ranking_margin"] * kwargs["ranking_margin_factor"]

        self.dropout_adj: float = kwargs["dropout_adj"]
        # transformer conv
        self.heads: int = kwargs["heads"]
        self.concat: int = kwargs["concat"]
        self.beta: bool = kwargs["beta"]
        self.dropout: float = kwargs["dropout"]
        self.edge_dim: int = kwargs["edge_dim"]
        self.root_weight: bool = kwargs["root_weight"]
        # optimizer
        self.learning_rate: float = kwargs["learning_rate"]
        self.weight_decay: float = kwargs["weight_decay"]

        # encoder
        self.encoder = nn.Sequential(collections.OrderedDict([
            ("enc_lin1", nn.Linear(self.input_dim, self.hidden_dim_mid, bias=False)),
            ("enc_lin1act", nn.SELU()),
            ("enc_dropout", nn.AlphaDropout(p=self.dropout)),
            ("enc_lin2", nn.Linear(self.hidden_dim_mid, self.hidden_dim, bias=False)),
            ("enc_lin2act", nn.SELU())
        ]))
        # message passing
        intermediate_dim: int = self.hidden_dim * (self.heads if self.concat else 1)
        self.message_passing = Sequential('x, edge_index, edge_attr, batch, is_training', collections.OrderedDict([
            ("mp_dropout_adj", (lambda edge_index, edge_attr, is_training: dropout_adj(edge_index, edge_attr=edge_attr,
                                                                                       p=self.dropout_adj,
                                                                                       training=is_training),
                                'edge_index, edge_attr, is_training -> tmp_edge_index, tmp_edge_attr')),
            ("mp_conv1", (TransformerConv(self.hidden_dim, self.hidden_dim, heads=self.heads, bias=False,
                                          concat=self.concat, edge_dim=self.edge_dim, dropout=self.dropout,
                                          beta=self.beta, root_weight=self.root_weight),
                          'x, tmp_edge_index, tmp_edge_attr -> x')),
            ("mp_conv1act", (nn.SELU(), 'x -> x')),
            ("mp_conv1dropout", (nn.AlphaDropout(p=self.dropout), 'x -> x')),
            ("mp_lin1", (nn.Linear(intermediate_dim, self.hidden_dim, bias=False), 'x -> x')),
            ("mp_lin1act", (nn.SELU(), 'x -> x')),
            ("mp_return", (lambda x, tmp_edge_index: (x, tmp_edge_index),
                           'x, tmp_edge_index -> x, tmp_edge_index'))
        ]))
        # decoder
        self.decoder = nn.Sequential(collections.OrderedDict([
            ("dec_lin1", nn.Linear(self.hidden_dim, self.hidden_dim_mid, bias=False)),
            ("dec_lin1act", nn.SELU()),
            ("dec_dropout", nn.AlphaDropout(p=self.dropout)),
            ("dec_lin2", nn.Linear(self.hidden_dim_mid, self.input_dim, bias=False)),
            ("dec_lin2act", nn.Sigmoid())
        ]))
        # simple linear transformation to obtain logits for classification of benchmark types
        self.classifier = nn.Linear(self.hidden_dim, self.output_dim, bias=True)
        # non-linear transformation to obtain logits for detection of anomalous executions
        chaos_dim: int = self.output_dim + self.hidden_dim + self.predecessor_dim
        self.cls_chaos1 = nn.Linear(chaos_dim, chaos_dim, bias=True)
        self.cls_chaos2 = nn.Linear(chaos_dim, 1, bias=True)

        # init weights / biases of linear layers
        self.apply(init_weights)
        # init bias of last linear layer differently (to account for class imbalance)
        nn.init.constant_(self.cls_chaos2.bias, torch.log(self.next_pos_neg_ratio).item())

        ### init metrics ###
        for phase, (name, clazz) in list(itertools.product(["train", "val", "test", "predict"],
                                                           [
                                                               ("chaos", torchmetrics.MetricCollection),
                                                               ("this_acc", torchmetrics.Accuracy),
                                                               ("next_acc", torchmetrics.Accuracy),
                                                               ("this_mse", torchmetrics.MeanSquaredError),
                                                               ("next_mse", torchmetrics.MeanSquaredError)
                                                           ])):
            attribute_name: str = "_".join([phase, name])
            clazz_args = []
            clazz_kwargs = {}
            if attribute_name.endswith("chaos"):
                clazz_args = [{
                    "acc": torchmetrics.Accuracy(), "rec": torchmetrics.Recall(),
                    "pre": torchmetrics.Precision(), "f1s": torchmetrics.F1Score(),
                }]
                clazz_kwargs = {"prefix": f"{attribute_name}_", "compute_groups": True}
            setattr(self, attribute_name, clazz(*clazz_args, **clazz_kwargs))
        ####################

    def forward(self, data):
        # 1. encode metric vectors
        enc = self.encoder(data.x)
        # 2. message passing of encodings to infer from context
        nxt, mod_e_index = self.message_passing(enc, data.edge_index,
                                                data.edge_attr, data.batch, self.training)
        
        # the first node of each graph has no precedessors and thus receives no information --> exclusion
        # some nodes might have too few precedessors --> exclusion
        # in case of dropout-adj, some nodes might have no precedessors as well --> exclusion
        counts = collections.Counter(mod_e_index[1, mod_e_index[0] != mod_e_index[1]].tolist())
        valid_node_indices = [counts[idx] >= data.min_predecessors.max().item() for idx in range(len(data.x))]
        # now: what are "valid" nodes and not chaos benchmark executions?
        valid_node_mask: torch.BoolTensor = torch.tensor(valid_node_indices).to(data.chaos)
        not_chaos_mask: torch.BoolTensor = torch.logical_and(valid_node_mask, ~data.chaos)

        # simplify downstream evaluation
        nxt = (nxt * valid_node_mask.to(torch.long)[:, None]) + \
              (enc.detach().clone() * (~valid_node_mask).to(torch.long)[:, None])
        
        # 3. classify
        enc_cls = self.classifier(enc)
        nxt_cls = self.classifier(nxt)

        # 4. get indication of anomaly-likeliness
        enc_norm = torch.linalg.vector_norm(enc, ord=GeneralConfig.vector_norm_ord, dim=-1, keepdim=False)
        nxt_norm = torch.linalg.vector_norm(nxt, ord=GeneralConfig.vector_norm_ord, dim=-1, keepdim=False)

        chaos_logits = self.cls_chaos2(F.selu(self.cls_chaos1(torch.cat([enc - nxt,
                                                                         data.onehot_predecessors,
                                                                         data.onehot], dim=-1)))).reshape(-1)
        # simplify downstream evaluation
        chaos_logits = (chaos_logits * valid_node_mask.to(torch.long)[:]) + \
              (torch.logit(data.chaos.detach().clone().to(torch.long), eps=0.01) * (~valid_node_mask).to(torch.long)[:])

        # simplify downstream evaluation
        nxt = (nxt * not_chaos_mask.to(torch.long)[:, None]) + \
              (enc.detach().clone() * (~not_chaos_mask).to(torch.long)[:, None])
        
        # 5. decode
        enc_dec = self.decoder(enc)
        nxt_dec = self.decoder(nxt)
        
        return enc, enc_norm, enc_dec, enc_cls, nxt, nxt_norm, nxt_dec, nxt_cls, chaos_logits

    @torch.no_grad()
    def configure_optimizers(self):
        """
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases).
        We are then returning the PyTorch optimizer object.
        Adapted from: https://github.com/karpathy/minGPT/blob/master/mingpt/model.py
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, torch_geometric.nn.Linear,)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn  # full param name
                # random note: because named_modules and named_parameters are recursive
                # we will see the same tensors p many times. but doing it this way
                # allows us to know which parent module any tensor p belongs to...
                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params),)
        assert len(
            param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params),)

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": self.weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.Adam(optim_groups, lr=self.learning_rate)
        return optimizer

    def training_step(self, data, batch_idx):
        with torch.no_grad():
            data.x += torch.normal(0, 0.01, size=data.x.shape).to(data.x) # add some noise
            data.edge_attr += torch.normal(0, 0.01, size=data.edge_attr.shape).to(data.edge_attr) # add some noise
        result_dict = self.calculate_batch_sores("train", data)
        result_dict["loss"] = result_dict["train_loss"]
        return result_dict

    def validation_step(self, data, batch_idx):
        return self.calculate_batch_sores("val", data)
        
    def validation_epoch_end(self, outputs):
        """ Necessary because of ray-tune integration."""
        result_dict = {f"ptl/{k}":v for k, v in pd.DataFrame(outputs).mean(axis=0).to_dict().items()}
        self.log_dict(result_dict, prog_bar=False, logger=False)
        self.logger.agg_and_log_metrics(result_dict, step=self.current_epoch)

    def test_step(self, data, batch_idx):
        return self.calculate_batch_sores("test", data)

    def calculate_batch_sores(self, name, data):
        class_clustering_loss = losses.MultipleLosses(
            [losses.TripletMarginLoss(distance=CosineSimilarity())],
            [miners.TripletMarginMiner(distance=CosineSimilarity())]
        )

        enc, enc_norm, enc_dec, enc_cls, nxt, nxt_norm, nxt_dec, nxt_cls, chaos_logits = self(data)

        chaos: torch.Tensor
        with torch.no_grad():
            chaos = torch.sigmoid(chaos_logits)
            chaos_pred_mod = torch.zeros((len(chaos), 2)).to(chaos)
            chaos_pred_mod[chaos < 0.5, 0] = 1
            chaos_pred_mod[chaos >= 0.5, 1] = 1
        
        chaos_true = data.chaos.to(enc.dtype)
        pos_samples_divider: torch.Tensor = getattr(chaos_true, "sum" if data.chaos.any() else "numel")()
        # used for loss computation
        loss_terms_dict: dict = {
            # focal loss, actually used. As proposed in one paper, normalize by number of positive samples (if any)
            f"{name}_loss_chaos_foc": O.sigmoid_focal_loss(chaos_logits, chaos_true,
                                                           gamma=self.focal_gamma,
                                                           alpha=self.focal_alpha,
                                                           reduction='sum') / pos_samples_divider,
            f"{name}_loss_this_ce": F.cross_entropy(enc_cls, data.bm_id),
            f"{name}_loss_next_ce": F.cross_entropy(nxt_cls, data.bm_id),
            f"{name}_loss_this_mse": F.mse_loss(enc_dec, data.x),
            f"{name}_loss_next_mse": F.mse_loss(nxt_dec, data.x),
            f"{name}_loss_cls_clst": class_clustering_loss(
                torch.cat([enc, nxt], dim=0),
                data.bm_id.repeat(2)),
            f"{name}_loss_this_normal_rkg": F.margin_ranking_loss(enc_norm[data.ranking_indices_all_normal[:, 0]] * data.ranking_factors_all_normal,
                                                                  enc_norm[data.ranking_indices_all_normal[:, 1]],
                                                                  data.ranking_targets_all_normal,
                                                                  margin=0),
            f"{name}_loss_next_normal_rkg": F.margin_ranking_loss(nxt_norm[data.ranking_indices_all_normal[:, 0]] * data.ranking_factors_all_normal,
                                                                  nxt_norm[data.ranking_indices_all_normal[:, 1]],
                                                                  data.ranking_targets_all_normal,
                                                                  margin=0)
        }
        # the following loss term only applies when chaos is present
        if len(data.ranking_indices_all_chaos):
            loss_terms_dict[f"{name}_loss_this_chaos_rkg"] = F.margin_ranking_loss(
                enc_norm[data.ranking_indices_all_chaos[:, 0]] * data.ranking_factors_all_chaos,
                enc_norm[data.ranking_indices_all_chaos[:, 1]],
                data.ranking_targets_all_chaos,
                margin=self.margin)
        
        to_log_dict = {
            f"{name}_loss": sum(list(loss_terms_dict.values())),
            f"{name}_chaos": (chaos_pred_mod, data.chaos.to(torch.long)),
            f"{name}_this_acc": (enc_cls, data.bm_id),
            f"{name}_next_acc": (nxt_cls, data.bm_id),
            f"{name}_this_mse": (enc_dec, data.x),
            f"{name}_next_mse": (nxt_dec, data.x)
        }
        
        to_return_dict = {}
        for m_name, m_value in to_log_dict.items():
            log_dict, batch_size = {m_name: m_value}, len(enc)
            if isinstance(m_value, tuple):
                m_cls_inst = getattr(self, m_name)
                out = m_cls_inst(*m_value)
                log_dict, batch_size = out if isinstance(out, dict) else {m_name: out}, len(m_value[0])
            self.log_dict(log_dict, prog_bar=False, on_epoch=True, batch_size=batch_size)
            to_return_dict = {**to_return_dict, **log_dict}
            
        return to_return_dict
       