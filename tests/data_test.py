from mininlp.data import EmbeddingsSource, encode_text, spacy_pipeline_cache
import pytest


@pytest.mark.slow
def test_encode_text():
    corpus = [
        "The 1st sequence.",
        "This is the second document. Make it a bit longer to be sure it doesn't have padding.",
        "The third sequence, finally.",
        "Small",
        "Small OK"
    ]
    custom_result = encode_text(corpus, EmbeddingsSource.Std)
    custom_result_sl = encode_text(corpus, EmbeddingsSource.Std, min_seq_len=2)
    bert_result = encode_text(corpus, EmbeddingsSource.DistilBert)
    bert_result_sl = encode_text(corpus, EmbeddingsSource.DistilBert, min_seq_len=3)

    nlp = spacy_pipeline_cache.get('en_core_web_lg')
    minimum_expected_tokens = [
        "The", "1st", "sequence.", "This", "is", "second", "document.", "Make", "it", "a", "bit", "longer."
        "third", "Small", "OK", " "
    ]
    expected_tokens = [token.text for doc in corpus for token in nlp(doc) if token.has_vector] + [" "]

    assert len(custom_result.vocab) >= len(minimum_expected_tokens)
    assert set(custom_result.vocab.idx_to_word) == set(expected_tokens)
    assert custom_result.X_padded[0][0] == custom_result.X_padded[2][0]
    # It's not expected than any of {"This", "is"} is further subdivided, so we can
    # check if the id of the second token of the second sequence is right
    assert custom_result.X_padded[1][1] == custom_result.vocab.word_to_idx['is']
    assert custom_result.vocab.pad_idx in custom_result.X_padded[0]
    assert custom_result.vocab.pad_idx not in custom_result.X_padded[1]
    assert custom_result.vocab.pad_idx in custom_result.X_padded[2]
    assert custom_result.seq_lengths[1] > custom_result.seq_lengths[0]
    assert custom_result.seq_lengths[1] > custom_result.seq_lengths[2]
    assert len(custom_result.seq_lengths) == custom_result.X_padded.shape[0] == 5

    assert len(custom_result_sl.seq_lengths) == custom_result_sl.X_padded.shape[0] == 4

    assert len(bert_result.vocab) >= len(minimum_expected_tokens)
    assert bert_result.X_padded[0][0] == bert_result.X_padded[2][0]
    # It's not expected than any of {"This", "is"} is further subdivided, so we can
    # check if the id of the third token of the second sequence is right (the first 
    # token is always [CLS])
    assert bert_result.X_padded[1][2] == bert_result.vocab.word_to_idx['is']
    assert bert_result.vocab.pad_idx in bert_result.X_padded[0]
    assert bert_result.vocab.pad_idx not in bert_result.X_padded[1]
    assert bert_result.vocab.pad_idx in bert_result.X_padded[2]
    assert bert_result.seq_lengths[1] > bert_result.seq_lengths[0]
    assert bert_result.seq_lengths[1] > bert_result.seq_lengths[2]
    assert len(bert_result.seq_lengths) == bert_result.X_padded.shape[0] == 5

    assert len(bert_result_sl.seq_lengths) == bert_result_sl.X_padded.shape[0] == 4
