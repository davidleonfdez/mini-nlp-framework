from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
import numpy as np
from sklearn.model_selection import KFold
import spacy
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer
from transformers.tokenization_utils import PreTrainedTokenizer
from typing import Dict, Iterable, List, Tuple, Union


DEFAULT_BS = 64


"""Vocabulary that maps tokens to ids and viceversa.

Args
    word_to_idx: dictionary that assigns a id to every token.
    idx_to_word: list of tokens. The position of a token in the list is the id assigned to that token.
    pad_idx: index of the padding token.
"""
class Vocab:
    def __init__(self, word_to_idx:Dict[str, int], idx_to_word:List[str], pad_idx:int=0):
        self._word_to_idx = word_to_idx
        self._idx_to_word = idx_to_word
        self._pad_idx = pad_idx

    def __len__(self) -> int:
        return len(self.idx_to_word)

    @property
    def word_to_idx(self) -> Dict[str, int]:
        return self._word_to_idx

    @property
    def idx_to_word(self) -> List[str]:
        return self._idx_to_word

    @property
    def pad_idx(self) -> int:
        return self._pad_idx


class HFVocabAdapter(Vocab):
    """Vocabulary adapted from HuggingFace tokenizer.
    
    Args
        tokenizer: HuggingFace tokenizer that contains a vocab.
    """
    def __init__(self, tokenizer:PreTrainedTokenizer):
        self.tokenizer = tokenizer
        self._word_to_idx = None
        self._idx_to_word = None

    @property
    def word_to_idx(self) -> Dict[str, int]:
        if self._word_to_idx is None:
            self._word_to_idx = self.tokenizer.get_vocab()
        return self._word_to_idx

    @property
    def idx_to_word(self) -> List[str]:
        if self._idx_to_word is None:
            word_to_idx = self.word_to_idx
            word_to_idx_sorted_by_idx = sorted(word_to_idx.items(), key=lambda item: item[1])
            min_idx = word_to_idx_sorted_by_idx[0][1]
            max_idx = word_to_idx_sorted_by_idx[-1][1]
            # We assume that there are no gaps in the index numbers, starting from 0,
            # and that every id has only one matching token
            assert (min_idx == 0) and (max_idx == len(word_to_idx) - 1)
            self._idx_to_word = [token for token, _ in word_to_idx_sorted_by_idx]
        return self._idx_to_word

    @property
    def pad_idx(self) -> int:
        return self.tokenizer.pad_token_id


class EmbeddingsSource(Enum):
    Std = 0
    Spacy = 1
    DistilBert = 2


class SpacyPipelineCache:
    "Cache of spaCy pipelines that can be queried by name"
    def __init__(self):
        self.cache = dict()

    def get(self, name:str):
        pipeline = self.cache.get(name, None)
        if pipeline is None:
            try:
                pipeline = spacy.load(name)
            except OSError:
                spacy.cli.download(name)
                pipeline = spacy.load(name)
            self.cache[name] = pipeline
        return pipeline


spacy_pipeline_cache = SpacyPipelineCache()


def get_kfolds(X:np.ndarray, y:np.ndarray, seq_lengths:Union[List[int], np.ndarray]=None, n:int=3) -> List[Tuple]:
    """
    Split `X`, `y` and `seq_lengths` into `n` folds and group them to form `n` train/test sets.

    Returns:
        List, each item is a tuple that contains a train and a test set. The training set has size 
        (total size) * (n-1)/n, while the test set size is (total size)/n.
        Let `out` be the return value, then each tuple `out[i]` contains:
            X_train: np.ndarray that contains the items of `X` assigned to the training set `i`.
            X_test: np.ndarray that contains the items of `X` assigned to the test set `i`.
            y_train: np.ndarray that contains the items of `y` assigned to the training set `i`.
            y_test: np.ndarray that contains the items of `y` assigned to the test set `i`.
            sl_train: np.ndarray that contains the items of `seq_lengths` assigned to the training set `i`. It's None
                when `seq_lengths is None`.
            sl_test: np.ndarray that contains the items of `seq_lengths` assigned to the test set `i`. It's None
                when `seq_lengths is None`.
    """
    kf = KFold(n_splits=n, random_state=None, shuffle=False)
    splits = []
    if isinstance(seq_lengths, list): seq_lengths = np.array(seq_lengths)
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        sl_train, sl_test = ((seq_lengths[train_index], seq_lengths[test_index]) if seq_lengths is not None
                             else (None, None))
        splits.append((X_train, X_test, y_train, y_test, sl_train, sl_test))
    return splits


def get_dl_from_tensors(x, y, *extra_xs, bs=DEFAULT_BS):
    dataset = TensorDataset(x, y, *extra_xs)
    return DataLoader(dataset, batch_size=bs)


@dataclass
class DataLoaders:
    train:DataLoader=None
    valid:DataLoader=None


def pad(sequences:List[List[str]], max_len:int, pad_idx:int):
    """Add `pad_idx` values to every List contained in `sequences` until reaching `max_len`.
    
    The update is performed inplace.
    """
    for seq in sequences:
        seq.extend([pad_idx] * (max_len - len(seq)))
    return sequences


@dataclass
class TextEncodingResult:
    X_padded:np.ndarray
    vocab:Vocab
    seq_lengths:List[int]


class TextEncoder(ABC):
    "Child classes must define a text encoding method"
    @abstractmethod
    def encode(text_list:Iterable[str], vocab:Vocab=None, min_seq_len:int=0) -> TextEncodingResult:
        """Tokenize and numericalize a list of text sequences. 
        Args:
            text_list: list of text documents to encode as a sequence of token ids.
            vocab: vocabulary used to convert from tokens to ids. If `None`, a vocabulary is built and returned as
                the `vocab` attribute of the output.
            min_seq_len: minimum accepted sequence length. The sequences with a length lower than `min_seq_len` are
                discarded.
        Returns:
            TextEncodingResult
                X_padded: array of shape [`len(corpus)`, max sequence length] containing the numericalized sequences
                    right-padded to the length of the longest sequence.
                vocab: the `vocab` parameter if it's not `None`; else, a vocabulary derived from `text_list` or
                    fixed (previously built), depending on the child class implementation.
                seq_length: the element `i` contains the actual length (without padding) of the sequence `X_padded[i]`.
        """


class CustomTextEncoder(TextEncoder):
    """Custom tokenizer and numericalizer.
    
    When `use_spacy_tokenizer is False`, the tokens are extracted by splitting the input sequences using whitespace
    as token delimiter.

    Args:
        use_spacy_tokenizer: whether to use a spaCy tokenizer.
    """
    def __init__(self, use_spacy_tokenizer=True, ignore_oov_terms=True):
        self.use_spacy_tokenizer = use_spacy_tokenizer
        self.ignore_oov_terms = ignore_oov_terms

    def encode(self, text_list:Iterable[str], vocab:Vocab=None, min_seq_len:int=0) -> TextEncodingResult:
        if self.use_spacy_tokenizer:
            nlp = spacy_pipeline_cache.get('en_core_web_lg')
            token_docs = [
                [t.text for t in doc if t.has_vector or (not self.ignore_oov_terms)] 
                for doc in nlp.pipe(text_list, n_process=-1)
            ]
        else:
            token_docs = [doc.split() for doc in text_list]
        if vocab is None:
            all_tokens = set([word for sentence in token_docs for word in sentence])
            if ' ' not in all_tokens:
                all_tokens.add(' ')
            word_to_idx = {token:idx for idx, token in enumerate(all_tokens)}
            pad_idx = word_to_idx[' ']
            vocab = Vocab(word_to_idx, list(all_tokens), pad_idx=pad_idx)
        # converting the docs to their token ids
        X = [
            [vocab.word_to_idx[token] for token in token_doc] 
            for token_doc in token_docs 
            if len(token_doc) >= min_seq_len
        ]
        seq_lengths = [len(xi) for xi in X]
        max_seq_len = max(seq_lengths)
        X_padded = np.array(pad(X, max_seq_len, vocab.pad_idx))
        return TextEncodingResult(X_padded, vocab, seq_lengths)


class HFTextEncoder(TextEncoder):
    """HuggingFace tokenizer and numericalizer.

    Args:
        checkpoint_name: name of the HuggingFace Hub checkpoint that contains the data needed to load the tokenizer.
    """
    def __init__(self, checkpoint_name):
        self.checkpoint_name = checkpoint_name

    def encode(self, text_list:Iterable[str], vocab:HFVocabAdapter=None, min_seq_len:int=0) -> TextEncodingResult:
        tokenizer = AutoTokenizer.from_pretrained(self.checkpoint_name)
        tokenized_data = tokenizer(text_list, padding=True)
        X_padded = np.array(tokenized_data['input_ids'])
        vocab = HFVocabAdapter(tokenizer)
        seq_lengths = (X_padded != tokenizer.pad_token_id).sum(axis=1)
        if min_seq_len > 0:
            min_sl_mask = seq_lengths > min_seq_len
            X_padded = X_padded[min_sl_mask]
            seq_lengths = seq_lengths[min_sl_mask]
        return TextEncodingResult(X_padded, vocab, seq_lengths.tolist())


def get_text_encoder_for(emb_source:EmbeddingsSource) -> TextEncoder:
    if emb_source == EmbeddingsSource.DistilBert:
        return HFTextEncoder('distilbert-base-uncased')
    return CustomTextEncoder()


def encode_text(corpus:Iterable[str], emb_source:EmbeddingsSource, **encode_kwargs) -> TextEncodingResult:
    """Prepare a list of texts to be used as inputs of a model that has an input embedding of type `emb_source`
    
    Args
        corpus: list of text sequences (probably full dataset) to encode.
        emb_source: source of the input embedding of the model you are encoding the text for.
    """
    encoder = get_text_encoder_for(emb_source)
    return encoder.encode(corpus, **encode_kwargs)
