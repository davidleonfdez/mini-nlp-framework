from collections import Counter
from mini_nlp_framework.data import Vocab
from mini_nlp_framework.layers import EmbeddingsSource, NormType
from mini_nlp_framework.models import (
    LinearClassifierFlattened, RNNClassifierFlattened, RNNClassifierMulti, SemiTransformerClfFlattened, 
    SemiTransformerClfMulti
)
from mini_nlp_framework.torch_utils import get_layers_of_type
import pytest
import torch
import torch.nn as nn


@pytest.mark.parametrize("n_classes", [2, 3])
@pytest.mark.parametrize("emb_source", [EmbeddingsSource.Std])
def test_lin_clf_flattened(n_classes, emb_source):
    idx_to_word = ['This', 'is', 'the', 'vocab', '<PAD>']
    word_to_idx = dict(zip(idx_to_word, range(len(idx_to_word))))
    vocab = Vocab(word_to_idx, idx_to_word)
    max_seq_len = 8
    emb_drop = 0.3
    lin_drop = 0.51
    simple_model = LinearClassifierFlattened(vocab, n_classes, max_seq_len, emb_source, emb_drop=emb_drop, lin_drop=lin_drop)
    n_simple_model_lin_layers = len(list(get_layers_of_type(simple_model, nn.Linear)))
    n_simple_model_lin_layers_in_emb = len(list(get_layers_of_type(simple_model.embedding, nn.Linear)))
    simple_model_dropout_layers = get_layers_of_type(simple_model, nn.Dropout)
    simple_model_dropout_ps_count = Counter(l.p for l in simple_model_dropout_layers)

    two_steps_lin_model = LinearClassifierFlattened(
        vocab, n_classes, max_seq_len, emb_source, emb_drop=emb_drop, lin_drop=lin_drop, two_steps_lin=True
    )
    n_two_steps_lin_model_lin_layers = len(list(get_layers_of_type(two_steps_lin_model, nn.Linear)))
    n_two_steps_lin_model_lin_layers_in_emb = len(list(get_layers_of_type(two_steps_lin_model.embedding, nn.Linear)))
    two_steps_lin_model_dropout_layers = get_layers_of_type(two_steps_lin_model, nn.Dropout)
    two_steps_lin_model_dropout_ps_count = Counter(l.p for l in two_steps_lin_model_dropout_layers)

    two_resblocks_model = LinearClassifierFlattened(
        vocab, n_classes, max_seq_len, emb_source, emb_drop=emb_drop, lin_drop=lin_drop, res_blocks_ftrs=[4, 1]
    )
    two_resblocks_model_dropout_layers = list(get_layers_of_type(two_resblocks_model, nn.Dropout))
    two_resblocks_model_head_dropout_layers = list(get_layers_of_type(two_resblocks_model.clf, nn.Dropout))
    two_resblocks_model_dropout_ps_count = Counter(l.p for l in two_resblocks_model_dropout_layers)
    two_resblocks_model_lin_layers = set(get_layers_of_type(two_resblocks_model, nn.Linear))
    two_resblocks_model_lin_layers_in_emb = set(get_layers_of_type(two_resblocks_model.embedding, nn.Linear))
    two_resblocks_model_lin_layers_not_in_emb = two_resblocks_model_lin_layers - two_resblocks_model_lin_layers_in_emb
    two_resblocks_model_lin_layers_not_in_emb_out_ftrs = Counter(l.out_features for l in two_resblocks_model_lin_layers_not_in_emb)
    two_resblocks_model_emb_dim = two_resblocks_model.embedding.out_ftrs

    inp = torch.tensor([
        [0, 1, 3, 2, 4, 4, 4, 4],
        [1, 1, 1, 3, 2, 3, 2, 0]
    ], dtype=torch.long)
    bs = inp.shape[0]
    simple_model_out = simple_model(inp)
    two_steps_lin_model_out = two_steps_lin_model(inp)
    two_resblocks_lin_model_out = two_resblocks_model(inp)

    expected_out_ftrs = 1 if n_classes <= 2 else n_classes
    expected_out_shape = (bs, expected_out_ftrs)

    assert simple_model_out.shape == expected_out_shape
    assert n_simple_model_lin_layers == 1 + n_simple_model_lin_layers_in_emb
    assert simple_model_dropout_ps_count[0] == 1
    assert simple_model_dropout_ps_count[lin_drop] == 1

    assert two_steps_lin_model_out.shape == expected_out_shape
    assert n_two_steps_lin_model_lin_layers == 2 + n_two_steps_lin_model_lin_layers_in_emb
    assert two_steps_lin_model_dropout_ps_count[0] == 1
    assert two_steps_lin_model_dropout_ps_count[lin_drop] == 2

    assert two_resblocks_lin_model_out.shape == expected_out_shape
    assert len(two_resblocks_model_lin_layers) == 3 + len(two_resblocks_model_lin_layers_in_emb)
    assert len(two_resblocks_model_head_dropout_layers) == 2
    assert two_resblocks_model_dropout_ps_count[emb_drop] == 1
    assert two_resblocks_model_dropout_ps_count[lin_drop] == 2
    assert two_resblocks_model_lin_layers_not_in_emb_out_ftrs == Counter({
        two_resblocks_model_emb_dim * 4: 1, two_resblocks_model_emb_dim: 1, expected_out_ftrs: 1
    })


@pytest.mark.slow
@pytest.mark.parametrize("n_classes", [2, 3])
@pytest.mark.parametrize("emb_source", [EmbeddingsSource.Spacy, EmbeddingsSource.DistilBert])
def test_lin_clf_flattened_integration(n_classes, emb_source):
    test_lin_clf_flattened(n_classes, emb_source)


@pytest.mark.parametrize("n_classes", [2, 3])
@pytest.mark.parametrize("emb_source", [EmbeddingsSource.Std])
def test_rnn_clf_flattened(n_classes, emb_source):
    idx_to_word = ['This', 'is', 'the', 'vocab', '<PAD>']
    word_to_idx = dict(zip(idx_to_word, range(len(idx_to_word))))
    vocab = Vocab(word_to_idx, idx_to_word)
    hidden_ftrs = 50
    max_seq_len = 7
    # This is a workaround to count the dropout layers without interferences, because there could be additional 
    # non-customizable dropout layers in the `BaseEmbedding` `model.embedding`
    emb_drop = 0.211
    rnn_drop = 0.111
    lin_drop = 0.311
    simple_model = RNNClassifierFlattened(
        vocab, n_classes, hidden_ftrs, max_seq_len, emb_source, emb_drop=emb_drop, lin_drop=lin_drop, rnn_drop=rnn_drop
    )
    n_simple_model_lin_layers = len(list(get_layers_of_type(simple_model, nn.Linear)))
    n_simple_model_lin_layers_in_emb = len(list(get_layers_of_type(simple_model.embedding, nn.Linear)))
    simple_model_dropout_layers = get_layers_of_type(simple_model, nn.Dropout)
    simple_model_dropout_ps_count = Counter(l.p for l in simple_model_dropout_layers)

    two_steps_lin_model = RNNClassifierFlattened(
        vocab, n_classes, hidden_ftrs, max_seq_len, emb_source, emb_drop=emb_drop, lin_drop=lin_drop, 
        rnn_drop=rnn_drop, two_steps_lin=True
    )
    n_two_steps_lin_model_lin_layers = len(list(get_layers_of_type(two_steps_lin_model, nn.Linear)))
    n_two_steps_lin_model_lin_layers_in_emb = len(list(get_layers_of_type(two_steps_lin_model.embedding, nn.Linear)))
    two_steps_lin_model_dropout_layers = get_layers_of_type(two_steps_lin_model, nn.Dropout)
    two_steps_lin_model_dropout_ps_count = Counter(l.p for l in two_steps_lin_model_dropout_layers)

    two_resblocks_model = RNNClassifierFlattened(
        vocab, n_classes, hidden_ftrs, max_seq_len, emb_source, emb_drop=emb_drop, lin_drop=lin_drop, 
        rnn_drop=rnn_drop, res_blocks_ftrs=[4, 1]
    )
    n_two_resblocks_model_lin_layers = len(list(get_layers_of_type(two_resblocks_model, nn.Linear)))
    n_two_resblocks_model_lin_layers_in_emb = len(list(get_layers_of_type(two_resblocks_model.embedding, nn.Linear)))
    n_two_resblocks_model_head_dropout_layers = len(list(get_layers_of_type(two_resblocks_model.clf, nn.Dropout)))
    two_resblocks_model_dropout_layers = get_layers_of_type(two_resblocks_model, nn.Dropout)
    two_resblocks_model_dropout_ps_count = Counter(l.p for l in two_resblocks_model_dropout_layers)

    lstm_model = RNNClassifierFlattened(
        vocab, n_classes, hidden_ftrs, max_seq_len, emb_source, rnn_cls=nn.LSTM, emb_drop=emb_drop, lin_drop=lin_drop
    )

    inp = torch.tensor([
        [0, 1, 3, 2, 4, 4, 4],
        [1, 1, 1, 3, 2, 3, 2]
    ], dtype=torch.long)
    seq_lengths = torch.tensor([4, 7], dtype=torch.long)
    bs = inp.shape[0]
    simple_model_out = simple_model(inp, seq_lengths)
    two_steps_lin_model_out = two_steps_lin_model(inp, seq_lengths)
    two_resblocks_lin_model_out = two_resblocks_model(inp, seq_lengths)
    lstm_model_out = lstm_model(inp, seq_lengths)

    expected_out_ftrs = 1 if n_classes <= 2 else n_classes
    expected_out_shape = (bs, expected_out_ftrs)

    assert simple_model_out.shape == expected_out_shape
    assert n_simple_model_lin_layers == 1 + n_simple_model_lin_layers_in_emb
    assert simple_model_dropout_ps_count[lin_drop] == 1
    assert simple_model_dropout_ps_count[emb_drop] == 1
    assert simple_model_dropout_ps_count[rnn_drop] == 0

    assert two_steps_lin_model_out.shape == expected_out_shape
    assert n_two_steps_lin_model_lin_layers == 2 + n_two_steps_lin_model_lin_layers_in_emb
    assert two_steps_lin_model_dropout_ps_count[lin_drop] == 2
    assert two_steps_lin_model_dropout_ps_count[emb_drop] == 1
    assert two_steps_lin_model_dropout_ps_count[rnn_drop] == 0

    assert two_resblocks_lin_model_out.shape == expected_out_shape
    assert n_two_resblocks_model_lin_layers == 3 + n_two_resblocks_model_lin_layers_in_emb
    assert n_two_resblocks_model_head_dropout_layers == 2
    assert two_resblocks_model_dropout_ps_count[lin_drop] == 2
    assert two_resblocks_model_dropout_ps_count[emb_drop] == 1
    assert two_resblocks_model_dropout_ps_count[rnn_drop] == 0

    assert lstm_model_out.shape == expected_out_shape
    assert len(list(get_layers_of_type(lstm_model, nn.LSTM))) == 1


@pytest.mark.slow
@pytest.mark.parametrize("n_classes", [2, 3])
@pytest.mark.parametrize("emb_source", [EmbeddingsSource.Spacy, EmbeddingsSource.DistilBert])
def test_rnn_clf_flattened_integration(n_classes, emb_source):
    test_rnn_clf_flattened(n_classes, emb_source)


#TODO: distilbert
@pytest.mark.parametrize("n_classes", [2, 6])
@pytest.mark.parametrize("emb_source", [EmbeddingsSource.Std])
def test_rnn_clf_multi(n_classes, emb_source):
    idx_to_word = ['This', 'is', 'all', 'the', 'vocab', '<PAD>']
    word_to_idx = dict(zip(idx_to_word, range(len(idx_to_word))))
    vocab = Vocab(word_to_idx, idx_to_word)
    hidden_ftrs = 300
    max_seq_len = 9
    # This is a workaround to count the dropout layers without interferences, because there could be additional 
    # non-customizable dropout layers in the `BaseEmbedding` `model.embedding`
    emb_drop = 0.231
    rnn_drop = 0.171
    lin_drop = 0.391
    simple_model = RNNClassifierMulti(
        vocab, n_classes, hidden_ftrs, max_seq_len, emb_source, emb_drop=emb_drop, lin_drop=lin_drop, rnn_drop=rnn_drop
    )
    n_simple_model_lin_layers = len(list(get_layers_of_type(simple_model, nn.Linear)))
    n_simple_model_lin_layers_in_emb = len(list(get_layers_of_type(simple_model.embedding, nn.Linear)))
    simple_model_dropout_layers = get_layers_of_type(simple_model, nn.Dropout)
    simple_model_dropout_ps_count = Counter(l.p for l in simple_model_dropout_layers)

    ln_model = RNNClassifierMulti(
        vocab, n_classes, hidden_ftrs, max_seq_len, emb_source, emb_drop=emb_drop, lin_drop=lin_drop, 
        rnn_drop=rnn_drop, norm_type=NormType.Layer
    )

    two_resblocks_model = RNNClassifierMulti(
        vocab, n_classes, hidden_ftrs, max_seq_len, emb_source, emb_drop=emb_drop, lin_drop=lin_drop, 
        rnn_drop=rnn_drop, res_blocks_ftrs=[4, 1]
    )
    n_two_resblocks_model_lin_layers = len(list(get_layers_of_type(two_resblocks_model, nn.Linear)))
    n_two_resblocks_model_lin_layers_in_emb = len(list(get_layers_of_type(two_resblocks_model.embedding, nn.Linear)))
    n_two_resblocks_model_head_dropout_layers = len(list(get_layers_of_type(two_resblocks_model.clf, nn.Dropout)))
    two_resblocks_model_dropout_layers = get_layers_of_type(two_resblocks_model, nn.Dropout)
    two_resblocks_model_dropout_ps_count = Counter(l.p for l in two_resblocks_model_dropout_layers)

    lstm_model = RNNClassifierMulti(
        vocab, n_classes, hidden_ftrs, max_seq_len, emb_source, rnn_cls=nn.LSTM, emb_drop=emb_drop, lin_drop=lin_drop
    )

    expected_out_ftrs = 1 if n_classes <= 2 else n_classes
    weight_tied_model = RNNClassifierMulti(
        vocab, n_classes, hidden_ftrs, max_seq_len, emb_source, emb_drop=emb_drop, lin_drop=lin_drop, weight_tying=True
    )
    weight_tied_model_last_lin = next(
        l for l in get_layers_of_type(weight_tied_model, nn.Linear) if l.out_features == expected_out_ftrs
    )

    inp = torch.tensor([
        [0, 1, 3, 2, 4, 5, 5, 5, 5],
        [1, 1, 1, 3, 2, 3, 2, 0, 4]
    ], dtype=torch.long)
    seq_lengths = torch.tensor([5, 9], dtype=torch.long)
    bs = inp.shape[0]
    simple_model_out = simple_model(inp, seq_lengths)
    ln_model_out = ln_model(inp, seq_lengths)
    two_resblocks_lin_model_out = two_resblocks_model(inp, seq_lengths)
    lstm_model_out = lstm_model(inp, seq_lengths)
    weight_tied_model_out = weight_tied_model(inp, seq_lengths)

    expected_out_shape = (bs, max_seq_len, expected_out_ftrs)

    assert simple_model_out.shape == expected_out_shape
    assert n_simple_model_lin_layers == 1 + n_simple_model_lin_layers_in_emb
    assert simple_model_dropout_ps_count[lin_drop] == 1
    assert simple_model_dropout_ps_count[emb_drop] == 1
    assert simple_model_dropout_ps_count[rnn_drop] == 0

    assert ln_model_out.shape == expected_out_shape
    assert len(list(get_layers_of_type(ln_model, nn.LayerNorm))) == 1

    assert two_resblocks_lin_model_out.shape == expected_out_shape
    assert n_two_resblocks_model_lin_layers == 3 + n_two_resblocks_model_lin_layers_in_emb
    assert n_two_resblocks_model_head_dropout_layers == 2
    assert two_resblocks_model_dropout_ps_count[lin_drop] == 2
    assert two_resblocks_model_dropout_ps_count[emb_drop] == 1
    assert two_resblocks_model_dropout_ps_count[rnn_drop] == 0

    assert lstm_model_out.shape == expected_out_shape
    assert len(list(get_layers_of_type(lstm_model, nn.LSTM))) == 1

    can_tie_weights = (
        (len(vocab.idx_to_word) == expected_out_ftrs) 
        and (hidden_ftrs == weight_tied_model.backbone.embedding.out_ftrs)
    )
    assert weight_tied_model_out.shape == expected_out_shape
    assert (not can_tie_weights) or (
        weight_tied_model_last_lin.weight is weight_tied_model.backbone.embedding.input_embedding.weight
    )


@pytest.mark.slow
@pytest.mark.parametrize("n_classes", [2, 3])
@pytest.mark.parametrize("emb_source", [EmbeddingsSource.Spacy, EmbeddingsSource.DistilBert])
def test_rnn_clf_multi_integration(n_classes, emb_source):
    test_rnn_clf_multi(n_classes, emb_source)


@pytest.mark.parametrize("n_classes", [2, 3])
@pytest.mark.parametrize("emb_source", [EmbeddingsSource.Std])
def test_semitfm_clf_flattened(n_classes, emb_source):
    idx_to_word = ['This', 'is', 'the', 'vocab', '<PAD>']
    word_to_idx = dict(zip(idx_to_word, range(len(idx_to_word))))
    vocab = Vocab(word_to_idx, idx_to_word)
    max_seq_len = 7
    n_heads = 3
    n_layers = 2
    tfm_mlp_ftrs = 128
    emb_drop = 0.204
    tfm_drop = 0.104
    lin_drop = 0.304
    simple_model = SemiTransformerClfFlattened(
        vocab, n_classes, max_seq_len, emb_source, n_heads=n_heads, n_layers=n_layers, tfm_mlp_ftrs=tfm_mlp_ftrs,  
        emb_drop=emb_drop, tfm_drop=tfm_drop, lin_drop=lin_drop
    )
    simple_model_lin_layers = list(get_layers_of_type(simple_model, nn.Linear))
    n_simple_model_lin_layers_in_emb = len(list(get_layers_of_type(simple_model.embedding, nn.Linear)))
    n_simple_model_att_layers = len(list(get_layers_of_type(simple_model.backbone, nn.MultiheadAttention)))        
    simple_model_dropout_layers = get_layers_of_type(simple_model, nn.Dropout)
    simple_model_dropout_ps_count = Counter(l.p for l in simple_model_dropout_layers)
    simple_model_lin_layers_ftrs_count = Counter(l.out_features for l in simple_model_lin_layers)

    two_steps_lin_model = SemiTransformerClfFlattened(
        vocab, n_classes, max_seq_len, emb_source, n_heads=n_heads, n_layers=n_layers, tfm_mlp_ftrs=tfm_mlp_ftrs,  
        emb_drop=emb_drop, tfm_drop=tfm_drop, lin_drop=lin_drop, two_steps_lin=True
    )
    two_steps_lin_model_lin_layers = list(get_layers_of_type(two_steps_lin_model, nn.Linear))
    n_two_steps_lin_model_lin_layers_in_emb = len(list(get_layers_of_type(two_steps_lin_model.embedding, nn.Linear)))
    two_steps_lin_model_dropout_layers = get_layers_of_type(two_steps_lin_model, nn.Dropout)
    two_steps_lin_model_dropout_ps_count = Counter(l.p for l in two_steps_lin_model_dropout_layers)

    two_resblocks_model = SemiTransformerClfFlattened(
        vocab, n_classes, max_seq_len, emb_source, n_heads=n_heads, n_layers=n_layers, tfm_mlp_ftrs=tfm_mlp_ftrs,  
        emb_drop=emb_drop, tfm_drop=tfm_drop, lin_drop=lin_drop, res_blocks_ftrs=[4, 1]
    )
    two_resblocks_model_lin_layers = list(get_layers_of_type(two_resblocks_model, nn.Linear))
    n_two_resblocks_model_lin_layers_in_emb = len(list(get_layers_of_type(two_resblocks_model.embedding, nn.Linear)))
    two_resblocks_model_dropout_layers = get_layers_of_type(two_resblocks_model, nn.Dropout)
    two_resblocks_model_dropout_ps_count = Counter(l.p for l in two_resblocks_model_dropout_layers)

    causal_model = SemiTransformerClfFlattened(
        vocab, n_classes, max_seq_len, emb_source, n_heads=n_heads, n_layers=n_layers, tfm_mlp_ftrs=tfm_mlp_ftrs,  
        emb_drop=emb_drop, tfm_drop=tfm_drop, lin_drop=lin_drop, use_causal_mask=True
    )

    inp = torch.tensor([
        [0, 1, 3, 2, 4, 4, 4],
        [1, 1, 1, 3, 2, 3, 2]
    ], dtype=torch.long)
    seq_lengths = torch.tensor([4, 7], dtype=torch.long)
    bs = inp.shape[0]
    simple_model_out = simple_model(inp, seq_lengths)
    two_steps_lin_model_out = two_steps_lin_model(inp, seq_lengths)
    two_resblocks_lin_model_out = two_resblocks_model(inp, seq_lengths)
    causal_model_out = causal_model(inp, seq_lengths)

    expected_out_ftrs = 1 if n_classes <= 2 else n_classes
    expected_out_shape = (bs, expected_out_ftrs)

    assert simple_model_out.shape == expected_out_shape
    assert len(simple_model_lin_layers) == 1 + 3 * n_layers + n_simple_model_lin_layers_in_emb
    assert n_simple_model_att_layers == n_layers
    assert simple_model_lin_layers_ftrs_count[expected_out_ftrs] == 1
    assert simple_model_lin_layers_ftrs_count[tfm_mlp_ftrs] == n_layers
    assert simple_model_dropout_ps_count[emb_drop] == 1
    assert simple_model_dropout_ps_count[lin_drop] == 1
    assert simple_model_dropout_ps_count[tfm_drop] == 3 * n_layers

    assert two_steps_lin_model_out.shape == expected_out_shape
    assert len(two_steps_lin_model_lin_layers) == 2 + 3 * n_layers + n_two_steps_lin_model_lin_layers_in_emb
    assert two_steps_lin_model_dropout_ps_count[emb_drop] == 1
    assert two_steps_lin_model_dropout_ps_count[lin_drop] == 2
    assert two_steps_lin_model_dropout_ps_count[tfm_drop] == 3 * n_layers

    assert two_resblocks_lin_model_out.shape == expected_out_shape
    assert len(two_resblocks_model_lin_layers) == 3 + 3 * n_layers + n_two_resblocks_model_lin_layers_in_emb
    assert two_resblocks_model_dropout_ps_count[emb_drop] == 1
    assert two_resblocks_model_dropout_ps_count[lin_drop] == 2
    assert two_resblocks_model_dropout_ps_count[tfm_drop] == 3 * n_layers

    assert causal_model_out.shape == expected_out_shape


@pytest.mark.slow
@pytest.mark.parametrize("n_classes", [2, 3])
@pytest.mark.parametrize("emb_source", [EmbeddingsSource.Spacy, EmbeddingsSource.DistilBert])
def test_semitfm_clf_flattened_integration(n_classes, emb_source):
    test_semitfm_clf_flattened(n_classes, emb_source)


@pytest.mark.parametrize("n_classes", [2, 3])
@pytest.mark.parametrize("emb_source", [EmbeddingsSource.Std])
def test_semitfm_clf_multi(n_classes, emb_source):
    idx_to_word = ['This', 'is', 'the', 'vocab', '<PAD>']
    word_to_idx = dict(zip(idx_to_word, range(len(idx_to_word))))
    vocab = Vocab(word_to_idx, idx_to_word)
    max_seq_len = 8
    n_heads = 3
    n_layers = 2
    tfm_mlp_ftrs = 128
    emb_drop = 0.205
    tfm_drop = 0.105
    lin_drop = 0.305
    simple_model = SemiTransformerClfMulti(
        vocab, n_classes, max_seq_len, emb_source, n_heads=n_heads, n_layers=n_layers, tfm_mlp_ftrs=tfm_mlp_ftrs,  
        emb_drop=emb_drop, tfm_drop=tfm_drop, lin_drop=lin_drop
    )
    simple_model_lin_layers = list(get_layers_of_type(simple_model, nn.Linear))
    n_simple_model_lin_layers_in_emb = len(list(get_layers_of_type(simple_model.embedding, nn.Linear)))
    n_simple_model_att_layers = len(list(get_layers_of_type(simple_model.backbone, nn.MultiheadAttention)))        
    simple_model_dropout_layers = get_layers_of_type(simple_model, nn.Dropout)
    simple_model_dropout_ps_count = Counter(l.p for l in simple_model_dropout_layers)
    simple_model_lin_layers_ftrs_count = Counter(l.out_features for l in simple_model_lin_layers)

    two_resblocks_model = SemiTransformerClfMulti(
        vocab, n_classes, max_seq_len, emb_source, n_heads=n_heads, n_layers=n_layers, tfm_mlp_ftrs=tfm_mlp_ftrs,  
        emb_drop=emb_drop, tfm_drop=tfm_drop, lin_drop=lin_drop, res_blocks_ftrs=[4, 1]
    )
    two_resblocks_model_lin_layers = list(get_layers_of_type(two_resblocks_model, nn.Linear))
    n_two_resblocks_model_lin_layers_in_emb = len(list(get_layers_of_type(two_resblocks_model.embedding, nn.Linear)))
    two_resblocks_model_dropout_layers = get_layers_of_type(two_resblocks_model, nn.Dropout)
    two_resblocks_model_dropout_ps_count = Counter(l.p for l in two_resblocks_model_dropout_layers)

    causal_model = SemiTransformerClfMulti(
        vocab, n_classes, max_seq_len, emb_source, n_heads=n_heads, n_layers=n_layers, tfm_mlp_ftrs=tfm_mlp_ftrs,  
        emb_drop=emb_drop, tfm_drop=tfm_drop, lin_drop=lin_drop, use_causal_mask=True
    )

    inp = torch.tensor([
        [0, 1, 3, 2, 3, 4, 4, 4],
        [1, 1, 1, 3, 2, 3, 2, 0],
    ], dtype=torch.long)
    seq_lengths = torch.tensor([5, 8], dtype=torch.long)
    bs = inp.shape[0]
    simple_model_out = simple_model(inp, seq_lengths)
    two_resblocks_lin_model_out = two_resblocks_model(inp, seq_lengths)
    causal_model_out = causal_model(inp, seq_lengths)

    expected_out_ftrs = 1 if n_classes <= 2 else n_classes
    expected_out_shape = (bs, max_seq_len, expected_out_ftrs)

    assert simple_model_out.shape == expected_out_shape
    assert len(simple_model_lin_layers) == 1 + 3 * n_layers + n_simple_model_lin_layers_in_emb    
    assert n_simple_model_att_layers == n_layers
    assert simple_model_dropout_ps_count[emb_drop] == 1
    assert simple_model_dropout_ps_count[lin_drop] == 1
    assert simple_model_dropout_ps_count[tfm_drop] == 3 * n_layers
    assert simple_model_lin_layers_ftrs_count[expected_out_ftrs] == 1
    assert simple_model_lin_layers_ftrs_count[tfm_mlp_ftrs] == n_layers
    
    assert two_resblocks_lin_model_out.shape == expected_out_shape
    assert len(two_resblocks_model_lin_layers) == 3 + 3 * n_layers + n_two_resblocks_model_lin_layers_in_emb
    assert two_resblocks_model_dropout_ps_count[emb_drop] == 1
    assert two_resblocks_model_dropout_ps_count[lin_drop] == 2
    assert two_resblocks_model_dropout_ps_count[tfm_drop] == 3 * n_layers

    assert causal_model_out.shape == expected_out_shape


@pytest.mark.slow
@pytest.mark.parametrize("n_classes", [2, 3])
@pytest.mark.parametrize("emb_source", [EmbeddingsSource.Spacy, EmbeddingsSource.DistilBert])
def test_semitfm_clf_multi_integration(n_classes, emb_source):
    test_semitfm_clf_multi(n_classes, emb_source)
