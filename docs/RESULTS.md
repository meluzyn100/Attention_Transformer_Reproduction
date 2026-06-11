# Results

| Model | Checkpoint | BLEU (sacrebleu) | Uwagi |
|---|---|---:|---|
| Base EN-DE | `checkpoints/base_en_de/averaged.pt` | 5.45 | hypotheses written to `results/newstest2014_base.txt` |
| Base EN-DE | `checkpoints/base_en_de_v2/step_039062.pt` | 3.67 | representative single checkpoint for comparison |

Reference target from the paper is around 27.3 BLEU for Base EN-DE; the current reproduction is still far below that baseline.