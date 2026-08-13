from iloptimus.core.sft import tokenize_sft_rows


class _Tokenizer:
    def apply_chat_template(self, messages, **_kwargs):
        if len(messages) == 1:
            return list(range(20))
        completion_length = len(messages[-1]["content"].split())
        return list(range(20 + completion_length))


def test_tokenization_is_eager_and_reports_completion_retention():
    rows = [
        {"prompt": "short", "completion": " ".join(["code"] * 40)},
        {"prompt": "long", "completion": " ".join(["code"] * 100)},
    ]
    dataset, stats = tokenize_sft_rows(rows, _Tokenizer(), max_seq_length=80)
    assert len(dataset) == 2
    assert len(dataset[0][0]) == 60
    assert stats["fully_retained_rows"] == 1
    assert stats["completion_tokens"] == 140
    assert stats["retained_completion_tokens"] == 100
    assert stats["completion_retention"] == 0.7143
