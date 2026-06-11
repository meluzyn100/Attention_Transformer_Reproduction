from __future__ import annotations

from collections.abc import Sequence

from sacrebleu import corpus_bleu


def compute_bleu(hypotheses: Sequence[str], references: Sequence[str]) -> float:
    if len(hypotheses) != len(references):
        raise ValueError("hypotheses and references must have the same length")

    return float(corpus_bleu(list(hypotheses), [list(references)]).score)
