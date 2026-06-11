from src.evaluation.metrics import compute_bleu


def test_compute_bleu_returns_score():
    score = compute_bleu(["the cat is on the mat"], ["the cat is on the mat"])
    assert score > 99.0
