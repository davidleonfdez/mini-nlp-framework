from abc import ABC, abstractmethod
from enum import Enum
import warnings
from mininlp.data import EmbeddingsSource, spacy_pipeline_cache, Vocab
from mininlp.misc import ProgressTracker
import spacy
import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm
from transformers import DistilBertModel
from typing import Callable, Optional


class Lambda(nn.Module):
    """Layer that wraps a function.

    One common use if to insert code in the middle of a nn.Sequential module without defining a custom module
    with its forward. For example:

    net = nn.Sequential(
        nn.Linear(10, 100),
        Lambda(lambda x: x * 5),
        nn.Linear(100, 1),
    )

    Args:
        func: callable that receives the input of the `forward` method and whose return value is then returned by
            `forward`.
    """
    def __init__(self, func:Callable[[torch.Tensor], torch.Tensor]):
        super().__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)


class BaseEmbedding(nn.Module):
    "Base class that every embedding module used in this library must subclass."
    @property
    def is_contextual(self) -> bool:
        """
        Child classes whose embedding vectors are dependant on context must return `True`.
        
        This means we cannot precalculate the embeddings with only a vocab, but with full
        sequences.
        """
        return False

    @property
    def is_pretrained(self) -> bool:
        "Child classes must return True if they load pretrained embeddings."
        return False

    @property
    @abstractmethod
    def input_embedding(self) -> nn.Module:
        "Child classes must return the input embedding of the module that they wrap, tipically a `nn.Embedding`."

    @property
    @abstractmethod
    def out_ftrs(self) -> int:
        "Child classes must return the output size of the embeddings, i.e., the size of each 1d embedding vector."


class StdEmbedding(BaseEmbedding):
    """
    Embedding container that includes a nn.Embedding, a positional embedding and a dropout layer.

    Args:
        vocab: vocabulary containing the mappings between id's and tokens for the data than the embedding can accept.
        max_seq_len: maximun sequence length (second dimension of the input) that this layer can expect to receive.
        embedding_dim: size of every embedding vector.
        p_drop: `p` param of a `nn.Dropout` layer placed after the `nn.Embedding`.
        add_pos_emb: if `True`m a trainable positional embedding is added to every item of the output of the 
            Embedding layer; i.e., the same tensor is added to batch[0], batch[1], ...
    """
    def __init__(self, vocab:Vocab, max_seq_len:int, embedding_dim:int=300, p_drop:float=0, add_pos_emb=False):
        super().__init__()
        vocab_sz = len(vocab.idx_to_word)
        self._embedding = nn.Embedding(vocab_sz, embedding_dim)
        layers = [
            # If conversion is not needed, Tensor.long() does nothing
            Lambda(lambda x: x.long()),
            self._embedding,
        ]
        if add_pos_emb:
            layers.append(PositionalEmbedding(self._embedding.embedding_dim, max_seq_len))
        layers.append(nn.Dropout(p_drop))

        self.layers = nn.Sequential(*layers)

    @property
    def input_embedding(self) -> nn.Module:
        return self._embedding

    @property
    def out_ftrs(self) -> int:
        return self._embedding.embedding_dim

    def forward(self, x):
        return self.layers(x)


class PositionalEmbedding(nn.Module):
    """Learnable positional embedding that is added to every sequence of the input tensor.
    
    Args:
        in_ftrs: number of input features (last dimension of the input), tipically the embedding dim of the word
            embeddings.
        max_seq_len: maximum sequence length (second dimension of the input) that this layer can expect to receive.
    """
    def __init__(self, in_ftrs:int, max_seq_len:int):
        super().__init__()
        self.pos_emb = nn.Parameter(torch.randn(max_seq_len, in_ftrs, requires_grad=True))
        
    def forward(self, x):
        return x + self.pos_emb[:x.shape[1]]


class SpacyEmbedding(BaseEmbedding):
    """Container of `nn.Embedding` initialized with the outputs of `nlp` for the elements of the vocab.
    
    Args:
        vocab: vocabulary containing the mappings between id's and tokens for the data than the embedding can accept.
        nlp: spaCy language object used to obtain a feature vector for each token of `vocab` and initialize the
            `nn.Embedding`.
        max_seq_len: maximum sequence length (second dimension of the input) that this layer can expect to receive.
        p_drop: `p` param of a `nn.Dropout` layer placed after the `nn.Embedding`.
        add_pos_emb: if `True`m a trainable positional embedding is added to every item of the output of the 
            Embedding layer; i.e., the same tensor is added to batch[0], batch[1], ...
        progress_tracker: tracker of the progress of the load of the embedding that is notified after each token of the
            vocab is converted into a feature vector.
    """
    def __init__(self, vocab:Vocab, nlp:spacy.language.Language, max_seq_len:int, p_drop:float=0, add_pos_emb=False,
                 progress_tracker:Optional[ProgressTracker]=None):
        super().__init__()
        embeddings = []
        for i, token_doc in enumerate(nlp.pipe(vocab.idx_to_word, n_process=-1)):
            # What nlp.pipe returns is actually a list of docs, not tokens, but we don't want to further subdivide our tokens,
            # which may happen, so we get the doc vector directly, which is actually our word vector.
            embeddings.append(token_doc.vector)
            if progress_tracker is not None: 
                progress_tracker.update_progress(i)
        self._embedding = nn.Embedding.from_pretrained(torch.Tensor(embeddings), freeze=False)
        layers = [
             # If conversion is not needed, Tensor.long() does nothing
            Lambda(lambda x: x.long()),
            self._embedding,
        ]
        if add_pos_emb:
            layers.append(PositionalEmbedding(self._embedding.embedding_dim, max_seq_len))
        layers.append(nn.Dropout(p_drop))
        self.layers = nn.Sequential(*layers)

    @property
    def is_pretrained(self) -> bool:
        return True

    @property
    def input_embedding(self) -> nn.Module:
        return self._embedding

    @property
    def out_ftrs(self) -> int:
        return self._embedding.embedding_dim

    def forward(self, x):
        return self.layers(x)


class DistilBertEmbedding(BaseEmbedding):
    """Container of a pretrained DistilBert contextual embedding.

    Args:
        p_drop: `p` param of a `nn.Dropout` layer placed as the last submodule.
    """
    def __init__(self, p_drop:float=0):
        super().__init__()
        checkpoint_name = "distilbert-base-uncased"
        self.distilbert = DistilBertModel.from_pretrained(checkpoint_name)
        self.dropout = nn.Dropout(p_drop)

    @property
    def is_pretrained(self) -> bool:
        return True

    @property
    def input_embedding(self) -> nn.Module:
        return self.distilbert.embeddings.word_embeddings

    @property
    def is_contextual(self) -> bool:
        return True

    @property
    def out_ftrs(self) -> int:
        return self.distilbert.config.dim

    def forward(self, x):
        # If conversion is not needed, Tensor.long() does nothing
        x = self.distilbert(x.long()).last_hidden_state
        return self.dropout(x)


def get_embedding(source:EmbeddingsSource, vocab, max_seq_len, add_pos_emb=False, **emb_kwargs) -> BaseEmbedding:
    """Build an instance of the `BaseEmbedding` child class that corresponds to `source`.
    
    Args:
        vocab: vocabulary that contains the input ids that the embedding must accept.
        max_seq_len: maximum sequence length (second dimension of the input) that the returned module must accept.
        add_pos_emb: not applicable when `source == EmbeddingsSource.DistilBert`, in which case it must be True.

    The rest of the kwargs are forwarded to the `__init__` method of the corresponding `BaseEmbedding` subclass.
    """
    if source == EmbeddingsSource.Std:
        return StdEmbedding(vocab, max_seq_len, add_pos_emb=add_pos_emb, **emb_kwargs)
    if source == EmbeddingsSource.Spacy:
        nlp = spacy_pipeline_cache.get('en_core_web_lg')
        return SpacyEmbedding(vocab, nlp, max_seq_len=max_seq_len, add_pos_emb=add_pos_emb, **emb_kwargs)
    if source == EmbeddingsSource.DistilBert:
        if not add_pos_emb:
            warnings.warn("Disabling positional embeddings is not allowed for DistilBert embedding type")
        return DistilBertEmbedding(**emb_kwargs)


class NormType(Enum):
    No=0
    Batch=1
    Layer=2


def get_norm(norm_type:NormType, n_ftrs:int):
    "Create an instance of a norm layer of `norm_type` with `n_ftrs` input features."
    if norm_type == NormType.Batch:
        return nn.BatchNorm1d(n_ftrs)
    if norm_type == NormType.Layer:
        return nn.LayerNorm(normalized_shape=(n_ftrs,))
    return nn.Identity()


def apply_norm(norm_layer:nn.Module, x:torch.Tensor):
    "Apply normalization to a input `x` of size (batch size, sequence length, n ftrs)"
    if isinstance(norm_layer, nn.BatchNorm1d):
        x = x.transpose(1, 2)
    x = norm_layer(x)
    if isinstance(norm_layer, nn.BatchNorm1d):
        x = x.transpose(1, 2)
    return x


class LinResBlock(nn.Module):
    """Residual block with a linear layer in the inner path.

    If `out_ftrs` is greater than `in_ftrs`, the features missing in the identity are filled with zeros.
    If `out_ftrs` is lower than `in_ftrs`, the features of the identity with an index greater or equal than `in_ftrs`
    are discarded.

    Args:
        in_ftrs: number of input features.
        out_ftrs: number of output features.
        act: whether to include a ReLU activacion right after the linear layer.
        p_drop: `p` param of a dropout layer included at the end.
        sn: if True, spectral normalization is applied to the linear layer.
    """
    def __init__(self, in_ftrs, out_ftrs, act=True, p_drop=0, sn=False):
        super().__init__()
        self.in_ftrs, self.out_ftrs = in_ftrs, out_ftrs
        lin_layer = nn.Linear(in_ftrs, out_ftrs)
        if sn: lin_layer = spectral_norm(lin_layer)
        main_path_layers = [lin_layer]
        if act: main_path_layers.append(nn.ReLU())
        if p_drop > 0: main_path_layers.append(nn.Dropout(p_drop))
        self.main_path = nn.Sequential(*main_path_layers)
        
    def forward(self, x):
        out = self.main_path(x)
        if self.in_ftrs == self.out_ftrs:
            identity = x
        elif self.in_ftrs > self.out_ftrs:
            identity = x[..., :self.out_ftrs]
        else:
            identity = torch.cat([x, torch.zeros(*(x.size()[:-1]), self.out_ftrs - self.in_ftrs, device=x.device)], 
                                 axis=-1)
        return out + identity
