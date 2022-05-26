from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import partial
from mininlp.data import EmbeddingsSource, Vocab
from mininlp.layers import (
    apply_norm, BaseEmbedding, get_embedding, get_norm, Lambda, LinResBlock, PositionalEmbedding, NormType
)
from mininlp.losses import LossFunction, flat_binary_cross_entropy_loss, flat_cross_entropy_loss
from mininlp.train import ClipGradOptions
from mininlp.torch_utils import get_best_available_device, get_layers_of_type
import torch
import torch.nn as nn
from torch.optim import Optimizer
from typing import List, Optional, Tuple, Type


@dataclass
class ClassificationHeadSingleArch:
    p_drop:float=0.2
    res_blocks_ftrs:Optional[List[int]]=None
    two_steps_lin:bool=False


class ClassificationHeadSingle(nn.Module):
    """
    Classifier head that outputs a prediction for every batch item (first dim)

    Args:
        in_ftrs: number of input features (last dimension).
        n_classes: number of output classes, which is the size of the last dimension of the output logits.
        max_seq_len: maximum sequence length (second dimension) of inputs that the model can receive.
        arch:
            p_drop: `p` param of the dropout layers.
            res_blocks_ftrs: multipliers of the number of output features of the residual blocks to include before the 
                linear block(s) that reduce the features. Therefore, `len(res_blocks_ftrs)` must be the number of 
                residual blocks that you wish to include and `res_blocks_ftrs[-1]` must always be 1.
            two_steps_lin: make the final linear transformation in two steps, with ReLU+dropout in between. This can 
                provide stability.
    """
    def __init__(self, in_ftrs:int, n_classes:int, max_seq_len:int, arch:Optional[ClassificationHeadSingleArch]=None):
        super().__init__()
        out_ftrs = 1 if n_classes <= 2 else n_classes
        if arch is None: arch = ClassificationHeadSingleArch()
        if arch.res_blocks_ftrs is None: arch.res_blocks_ftrs = []
        arch.res_blocks_ftrs.insert(0, 1)
        def _rb_drop_value(i):
            return arch.p_drop if i < (len(arch.res_blocks_ftrs) - 1) else 0
        layers = [
            LinResBlock(
                in_ftrs * arch.res_blocks_ftrs[i-1], 
                in_ftrs * arch.res_blocks_ftrs[i], 
                p_drop=_rb_drop_value(i)
            ) 
            for i in range(1, len(arch.res_blocks_ftrs))
        ]
        if arch.two_steps_lin:
            if out_ftrs == 1:
                layers.extend([
                    nn.Dropout(arch.p_drop),
                    nn.Linear(in_ftrs, 1),
                    Lambda(lambda x: x.contiguous().view(x.shape[0], -1)),
                    nn.ReLU(),
                    nn.Dropout(arch.p_drop),
                    nn.Linear(max_seq_len, out_ftrs)
                ])
            else:
                # For multi-class classification it doesn't make sense to reduce the features to 1 first, so we
                # transpose to reduce to 1 just at the sequence dimension
                layers.extend([
                    nn.Dropout(arch.p_drop),
                    Lambda(lambda x: x.transpose(1, 2)),
                    nn.Linear(max_seq_len, 1),
                    Lambda(lambda x: x.contiguous().view(x.shape[0], -1)),
                    nn.ReLU(),
                    nn.Dropout(arch.p_drop),
                    nn.Linear(in_ftrs, out_ftrs)
                ])
        else:
            layers.extend([
                Lambda(lambda x: x.contiguous().view(x.shape[0], -1)),
                nn.Dropout(arch.p_drop),
                nn.Linear(in_ftrs * max_seq_len, out_ftrs),                
            ])
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


@dataclass
class ClassificationHeadMultiArch:
    p_drop:float=0
    res_blocks_ftrs:Optional[List[int]]=None


class ClassificationHeadMulti(nn.Module):
    """
    Classification head that outputs a prediction for every position of the input sequences.

    Args:
        in_ftrs: number of input features (last dimension).
        n_classes: number of output classes, which is the size of the last dimension of the output logits.
        arch:
            p_drop: `p` param of the dropout layer.
            res_blocks_ftrs: multipliers of the number of output features of the residual blocks that must be included 
                before the final dropout-linear. Therefore, `len(res_blocks_ftrs)` must be the number of residual blocks 
                that you wish to include and `res_blocks_ftrs[-1]` must always be 1.
    """
    def __init__(self, in_ftrs:int, n_classes:int, arch:Optional[ClassificationHeadMultiArch]=None):
        super().__init__()
        self.out_ftrs = 1 if n_classes <= 2 else n_classes
        if arch is None: arch = ClassificationHeadMultiArch()
        if arch.res_blocks_ftrs is None: arch.res_blocks_ftrs = []
        arch.res_blocks_ftrs.insert(0, 1)
        def _rb_drop_value(i):
            return arch.p_drop if i < (len(arch.res_blocks_ftrs) - 1) else 0
        res_blocks = [
            LinResBlock(
                in_ftrs * arch.res_blocks_ftrs[i-1], in_ftrs * arch.res_blocks_ftrs[i], p_drop=_rb_drop_value(i)
            )
            for i in range(1, len(arch.res_blocks_ftrs))
        ]
        self.layers = nn.Sequential(
            *res_blocks,
            nn.Dropout(arch.p_drop),            
            nn.Linear(in_ftrs, self.out_ftrs),
        )

    def forward(self, x):
        return self.layers(x)


class LinearClassifierFlattened(nn.Module):
    """
    Classifier composed by an embedding and a linear layer that outputs a single prediction for every batch item (1st dim).

    Args:
        vocab: vocabulary of the data
        n_classes: number of output classes, which is the size of the last dimension of the output logits
        max_seq_len: maximum sequence length of inputs that the model can receive
        embedding_source: type of embedding
        emb_drop: `p` param of the dropout layer placed after the embedding. Not applicable when 
            `res_blocks_ftrs` is None, because the dropout layer after the embedding is also right before the
            final linear layer, so only `lin_drop` is taken into account.
        head_arch:
            p_drop: `p` param of the dropout layers placed in the residual blocks and the one before the final linear 
                layer.
            res_blocks_ftrs: multipliers of the number of output features of the residual blocks to include before the
                linear block(s) that reduce the features. Therefore, `len(res_blocks_ftrs)` must be the number of 
                residual blocks that you wish to include and `res_blocks_ftrs[-1]` must always be 1.
            two_steps_lin: make the final linear transformation in two steps, with ReLU+dropout in between. This can 
                provide stability.
    """
    def __init__(
        self, vocab:Vocab, n_classes:int, max_seq_len:int, embedding_source:EmbeddingsSource, emb_drop:float=0.15,
        head_arch:Optional[ClassificationHeadSingleArch]=None
    ):
        super().__init__()
        if head_arch is None: head_arch = ClassificationHeadSingleArch()
        if (head_arch.res_blocks_ftrs is None) or (len(head_arch.res_blocks_ftrs) == 0):
            # In this case there would be 2 consecutive dropout layers if `emb_drop > 0`
            emb_drop = 0.
        self._embedding = get_embedding(embedding_source, vocab, max_seq_len, p_drop=emb_drop)
        emb_out_ftrs = self._embedding.out_ftrs
        self.clf = ClassificationHeadSingle(emb_out_ftrs, n_classes, max_seq_len, head_arch)
    
    @property
    def embedding(self) -> BaseEmbedding:
        return self._embedding

    def forward(self, x):
        return self.clf(self._embedding(x))


@dataclass
class RNNBackboneArch:
    embedding_source:EmbeddingsSource
    rnn_cls:Type[nn.RNNBase]=nn.RNN
    emb_drop:float=0.1
    rnn_drop:float=0.1


class RNNBackbone(nn.Module):
    """
    Backbone composed by an embedding and an RNN.

    Args:
        vocab: vocabulary of the data.
        hidden_ftrs: number of hidden features of the RNN.
        max_seq_len: maximum sequence length of inputs that the model can receive.
        arch:
            embedding_source: type of embedding.
            rnn_cls: class of the recurrent neural network. It can be nn.RNN or nn.LSTM.
            emb_drop: `p` param of the dropout layer placed after the embedding.
            rnn_drop: `p` param of the dropout applied on the outputs of each RNN layer except the last layer.
    """
    def __init__(self, vocab:Vocab, hidden_ftrs:int, max_seq_len:int, arch:RNNBackboneArch, **rnn_args):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.embedding = get_embedding(arch.embedding_source, vocab, max_seq_len, p_drop=arch.emb_drop)
        emb_out_ftrs = self.embedding.out_ftrs
        self.rnn = arch.rnn_cls(emb_out_ftrs, hidden_ftrs, batch_first=True, dropout=arch.rnn_drop, **rnn_args)

    def forward(self, x, seq_lengths):
        """
        Forward pass.

        Args:
            x: input tensor of size (batch size, seq length)
            seq_lenghts: long tensor of size (batch size,). It contains the length of every sequence
                in the batch
        Returns:
            Float tensor of size (batch size, seq length, hidden_ftrs)
        """
        emb = torch.nn.utils.rnn.pack_padded_sequence(
            self.embedding(x), seq_lengths, batch_first=True, enforce_sorted=False
        )
        rnn_out, _ = self.rnn(emb)
        rnn_out, _ = torch.nn.utils.rnn.pad_packed_sequence(
            rnn_out, batch_first=True, padding_value=0.0, total_length=self.max_seq_len
        )
        return rnn_out


class RNNClassifierFlattened(nn.Module):
    """
    Classifier composed by an embedding and an RNN that outputs a prediction for every batch item (first dim).

    Args:
        vocab: vocabulary of the data.
        n_classes: number of output classes, which is the size of the last dimension of the output logits.
        hidden_ftrs: number of hidden features used by the RNN.
        max_seq_len: maximum sequence length of inputs that the model can receive.
        backbone_arch:
            embedding_source: type of embedding.
            rnn_cls: class of the recurrent neural network. It can be nn.RNN or nn.LSTM.
            emb_drop: `p` param of the dropout layer placed after the embedding.
            rnn_drop: `p` param of the dropout applied on the outputs of each RNN layer except the last layer.
        head_arch:
            p_drop: `p` param of the dropout layer placed before the final linear layer.
            res_blocks_ftrs: multipliers of the number of output features of the residual blocks to include before the 
                linear block(s) that reduce the features. Therefore, `len(res_blocks_ftrs)` must be the number of residual 
                blocks that you wish to include and `res_blocks_ftrs[-1]` must always be 1.
            two_steps_lin: make the final linear transformation in two steps, with ReLU+dropout in between. This can 
                provide stability.
    """
    def __init__(
        self, vocab, n_classes:int, hidden_ftrs:int, max_seq_len:int, backbone_arch:RNNBackboneArch, 
        head_arch:Optional[ClassificationHeadSingleArch]=None, **rnn_args
    ):
        super().__init__()
        self.backbone = RNNBackbone(vocab, hidden_ftrs, max_seq_len, backbone_arch, **rnn_args)
        self.clf = ClassificationHeadSingle(hidden_ftrs, n_classes, max_seq_len, head_arch)
    
    @property
    def embedding(self) -> BaseEmbedding:
        return self.backbone.embedding

    def forward(self, x, seq_lengths):
        """
        Forward pass.

        Args:
            x: input tensor of size (batch size, seq length)
            seq_lenghts: long tensor of size (batch size,). It contains the length of every sequence
                in the batch
        Returns:
            Float tensor of size (batch size, n_classes)
        """
        x = self.backbone(x, seq_lengths)
        return self.clf(x)


class RNNClassifierMulti(nn.Module):
    """
    Classifier composed by an embedding and an RNN that outputs a prediction for every position of the input sequences.

    Args:
        vocab: vocabulary of the data
        n_classes: number of output classes, which is the size of the last dimension of the output logits. For a
            language model, it should be the length of the vocabulary.
        hidden_ftrs: number of hidden features used by the RNN.
        max_seq_len: maximum sequence length of inputs that the model can receive.
        backbone_arch:
            embedding_source: type of embedding.
            rnn_cls: class of the recurrent neural network. It can be nn.RNN or nn.LSTM.
            emb_drop: `p` param of the dropout layer placed after the embedding.
            rnn_drop: `p` param of the dropout applied on the outputs of each RNN layer except the last layer.
        head_arch:
            lin_drop: `p` param of the dropout layer placed before the final linear layer.
            res_blocks_ftrs: multipliers of the number of output features of the residual blocks that must be included 
                before the final dropout-linear. Therefore, `len(res_blocks_ftrs)` must be the number of residual blocks 
                that you wish to include and `res_blocks_ftrs[-1]` must always be 1.        
        weight_tying: if True, the weights of the input embedding and the weights of the
            final linear layer are the same.
        norm_type: type of normalization applied before the last dropout and linear layer.

    """
    def __init__(
        self, vocab:Vocab, n_classes, hidden_ftrs:int, max_seq_len:int, backbone_arch:RNNBackboneArch, 
        head_arch:Optional[ClassificationHeadMultiArch]=None, weight_tying=False, norm_type=NormType.No, 
        **rnn_args
    ):
        super().__init__()
        self.backbone = RNNBackbone(vocab, hidden_ftrs, max_seq_len, backbone_arch, **rnn_args)
        self.norm = get_norm(norm_type, hidden_ftrs)
        self.clf = nn.Sequential(
            Lambda(lambda x: apply_norm(self.norm, x)),
            ClassificationHeadMulti(hidden_ftrs, n_classes, head_arch),
        )
        out_ftrs = self.clf[-1].out_ftrs
        can_tie_weights = (hidden_ftrs == self.backbone.embedding.out_ftrs) and (out_ftrs == len(vocab.idx_to_word))
        if weight_tying and can_tie_weights:
            # TODO: I don't think it's ok with DistilBert
            self.clf[-1].layers[-1].weight = self.backbone.embedding.input_embedding.weight
    
    @property
    def embedding(self) -> BaseEmbedding:
        return self.backbone.embedding

    def forward(self, x, seq_lengths):
        """
        Forward pass.

        Args:
            x: input tensor of size (batch size, seq length).
            seq_lenghts: long tensor of size (batch size,). It contains the actual length of every sequence
                in the batch.
        Returns:
            Float tensor of size (batch size, max seq length, n_classes).
        """
        x = self.backbone(x, seq_lengths)
        return self.clf(x)


@dataclass
class SemiTransformerBackboneArch:
    embedding_source:EmbeddingsSource
    n_heads:int=1 
    n_layers:int=6 
    tfm_mlp_ftrs:int=2048
    emb_drop:float=0.1
    tfm_drop:float=0.1


class SemiTransformerBackbone(nn.Module):
    """
    Backbone composed by an embedding and a transformer encoder.

    Args:
        vocab: vocabulary of the data.
        max_seq_len: maximum sequence length of inputs that the model can receive. 
        arch:         
            embedding_source: type of embedding.    
            n_heads: number of heads in the attention layers.
            n_layers: number of blocks (Attention+MLP) of the transformer.
            tfm_mlp_ftrs: number of intermediate features used in the fully connected layers of the transformer.
            emb_drop: `p` param of the dropout layer placed after the embedding.
            tfm_drop: `p` param of the dropout layers placed inside the transformer.
    """
    def __init__(
            self, vocab, max_seq_len:int, arch:SemiTransformerBackboneArch, use_causal_mask=False, **tfm_enc_kwargs
        ):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.use_causal_mask = use_causal_mask
        self.embedding = get_embedding(
            arch.embedding_source, vocab, max_seq_len, add_pos_emb=True, p_drop=arch.emb_drop
        )
        in_ftrs = self.embedding.out_ftrs
        encoder_layer = nn.TransformerEncoderLayer(
            in_ftrs, arch.n_heads, dim_feedforward=arch.tfm_mlp_ftrs, dropout=arch.tfm_drop, batch_first=True, 
            **tfm_enc_kwargs,
        )
        encoder_norm = nn.LayerNorm(in_ftrs)
        self.encoder = nn.TransformerEncoder(encoder_layer, arch.n_layers, encoder_norm)
        self._reset_enc_params()
        
    def _reset_enc_params(self):
        for p in self.encoder.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x, seq_lengths):
        pad_mask = (torch.arange(x.shape[1])[None] >= seq_lengths[:, None]).to(x.device)
        att_mask = torch.ones(x.shape[1], x.shape[1]).triu(diagonal=1).bool().to(x.device) if self.use_causal_mask else None
        return self.encoder(self.embedding(x), src_key_padding_mask=pad_mask, mask=att_mask)


class SemiTransformerClfFlattened(nn.Module):
    """
    Classifier composed by an embedding + (transformer encoder) that outputs a prediction for every batch item (first dim).

    Args:
        vocab: vocabulary of the data
        n_classes: number of output classes, which is the size of the last dimension of the output logits
        max_seq_len: maximum sequence length of inputs that the model can receive 
        backbone_arch:          
            embedding_source: type of embedding
            n_heads: number of heads in the attention layers
            tfm_mlp_ftrs: number of intermediate features used in the fully connected layers of the transformer
            emb_drop: `p` param of the dropout layer placed after the embedding
            tfm_drop: `p` param of the dropout layers placed inside the transformer
        head_arch:
            lin_drop: `p` param of the dropout layer placed before the final linear layer
            res_blocks_ftrs: multipliers of the number of output features of the residual blocks to include before the
                linear block(s) that reduce the features. Therefore, `len(res_blocks_ftrs)` must be the number of 
                residual blocks that you wish to include and `res_blocks_ftrs[-1]` must always be 1.
            two_steps_lin: make the final linear transformation in two steps, with ReLU+dropout in between. This can 
                provide stability.
    """
    def __init__(
            self, vocab, n_classes, max_seq_len:int, backbone_arch:SemiTransformerBackboneArch, 
            head_arch:Optional[ClassificationHeadSingleArch]=None, use_causal_mask=False, **tfm_enc_kwargs
        ):
        super().__init__()
        self.backbone = SemiTransformerBackbone(
            vocab, max_seq_len, backbone_arch, use_causal_mask=use_causal_mask, **tfm_enc_kwargs,
        )
        hidden_ftrs = self.backbone.embedding.out_ftrs
        self.clf = ClassificationHeadSingle(hidden_ftrs, n_classes, max_seq_len, head_arch)
        
    @property
    def embedding(self) -> BaseEmbedding:
        return self.backbone.embedding

    def forward(self, x, seq_lengths):
        """
        Forward pass.

        Args:
            x: input tensor of size (batch size, seq length)
            seq_lenghts: long tensor of size (batch size,). It contains the length of every sequence
                in the batch
        Returns:
            Float tensor of size (batch size, n_classes)
        """
        return self.clf(self.backbone(x, seq_lengths)) 


class SemiTransformerClfMulti(nn.Module):
    """
    Classifier composed by an embedding + (transformer encoder) that outputs a pred for every pos. of the in sequences.

    Args:
        vocab: vocabulary of the data.
        n_classes: number of output classes, which is the size of the last dimension of the output logits.
        max_seq_len: maximum sequence length of inputs that the model can receive.
        backbone_arch:
            embedding_source: type of embedding.
            n_heads: number of heads in the attention layers.
            tfm_mlp_ftrs: number of intermediate features used in the fully connected layers of the transformer.
            emb_drop: `p` param of the dropout layer placed after the embedding.
            tfm_drop: `p` param of the dropout layers placed inside the transformer.
        head_arch:
            lin_drop: `p` param of the dropout layer placed before the final linear layer.
            res_blocks_ftrs: multipliers of the number of output features of the residual blocks that must be included
                before the final dropout-linear. Therefore, `len(res_blocks_ftrs)` must be the number of residual 
                blocks that you wish to include and `res_blocks_ftrs[-1]` must always be 1.
    """
    def __init__(
            self, vocab, n_classes, max_seq_len:int, backbone_arch:SemiTransformerBackboneArch, 
            head_arch:Optional[ClassificationHeadMultiArch]=None, use_causal_mask=False, 
            **tfm_enc_kwargs
        ):
        super().__init__()
        self.backbone = SemiTransformerBackbone(
            vocab, max_seq_len, backbone_arch, use_causal_mask=use_causal_mask, **tfm_enc_kwargs
        )
        hidden_ftrs = self.backbone.embedding.out_ftrs
        self.clf = ClassificationHeadMulti(hidden_ftrs, n_classes, head_arch)
        
    @property
    def embedding(self) -> BaseEmbedding:
        return self.backbone.embedding

    def forward(self, x, seq_lengths):
        """
        Forward pass.

        Args:
            x: input tensor of size (batch size, seq length).
            seq_lenghts: long tensor of size (batch size,). It contains the actual length of every sequence
                in the batch.
        Returns:
            Float tensor of size (batch size, max seq length, n_classes).
        """
        return self.clf(self.backbone(x, seq_lengths)) 


class BaseModelProvider(ABC):
    @abstractmethod
    def create(self, hp=None, device=None) -> Tuple[nn.Module, Optimizer, LossFunction, ClipGradOptions]:
        """
        Create a model, optimizer and loss function appropriate for a problem type specific to the child class.
        
        Args: 
            hp: hyperparameters needed to initialize the model and the optimizer.
            device: device with Pytorch format.
        Returns:
            Initialized model, optimizer and loss function.
        """


@dataclass
class ClfHyperParameters:
    lr:float=1e-3
    embedding_lr:float=5e-5
    wd:float=0
    adam_betas:Tuple[float,float]=(0.9, 0.999)


class QuickClassifierProvider(BaseModelProvider):
    "Provider of objects needed to train a simple classifier."
    embedding_source=EmbeddingsSource.Spacy

    def __init__(self, vocab:Vocab, max_seq_len:int, n_classes:int):
        self.vocab = vocab
        self.max_seq_len = max_seq_len
        self.n_classes = n_classes

    def create(self, hp:ClfHyperParameters=None, device=None) -> Tuple[
        nn.Module, Optimizer, LossFunction, ClipGradOptions
    ]:
        if hp is None: hp = ClfHyperParameters()
        if device is None: device = get_best_available_device()
        model = LinearClassifierFlattened(
            self.vocab, self.n_classes, self.max_seq_len, self.embedding_source, emb_drop=0.15,
        )
        model.to(device)
        embedding_params = model.embedding.parameters()
        clf_params = model.clf.parameters()
        opt = torch.optim.AdamW([
            {'params': clf_params},
            {'params': embedding_params, 'lr': hp.embedding_lr},
        ], lr=hp.lr, weight_decay=hp.wd, betas=hp.adam_betas)
        loss = flat_binary_cross_entropy_loss if self.n_classes <= 2 else flat_cross_entropy_loss
        clip_grad = None
        return model, opt, loss, clip_grad


@dataclass
class LMHyperParameters:
    lr:float=1e-3
    transformer_lr:float=1e-4
    embedding_lr:float=1e-4
    wd:float=0
    adam_betas:Tuple[float,float]=(0.9, 0.999)


class CustomLanguageModelProvider(BaseModelProvider):
    "Provider of objects needed to train a causal language model"
    embedding_source=EmbeddingsSource.Spacy

    def __init__(self, vocab:Vocab, max_seq_len:int):
        self.vocab = vocab
        self.max_seq_len = max_seq_len
    
    def create(self, hp:LMHyperParameters=None, device=None)-> Tuple[
        nn.Module, Optimizer, LossFunction, ClipGradOptions
    ]:
        if hp is None: hp = LMHyperParameters()
        if device is None: device = get_best_available_device()
        n_classes = len(self.vocab.idx_to_word)
        model = SemiTransformerClfMulti(
            self.vocab, 
            n_classes, 
            self.max_seq_len,
            SemiTransformerBackboneArch(self.embedding_source, n_heads=3, emb_drop=0.2, tfm_drop=0.2),
            use_causal_mask=True
        )
        # TODO: nn.parallel or ddp??
        model.to(device)
        # TODO: watchout with Distilbert, then I'd be losing the rest of the embedding parameters with this
        embedding_params = model.embedding.input_embedding.parameters()
        tfm_params = model.backbone.encoder.parameters()
        clf_params = model.clf.parameters()
        pos_emb_layer = next(get_layers_of_type(model, PositionalEmbedding))
        pos_emb_params = pos_emb_layer.parameters()
        opt = torch.optim.AdamW([
            {'params': clf_params},
            {'params': tfm_params, 'lr': hp.transformer_lr},
            {'params': embedding_params, 'lr': hp.embedding_lr},
            {'params': pos_emb_params, 'lr': hp.lr}
        ], lr=hp.lr, weight_decay=hp.wd, betas=hp.adam_betas)
        loss = partial(flat_cross_entropy_loss, ignore_index=self.vocab.pad_idx)
        clip_grad = ClipGradOptions(model.parameters(), 1.)
        return model, opt, loss, clip_grad
