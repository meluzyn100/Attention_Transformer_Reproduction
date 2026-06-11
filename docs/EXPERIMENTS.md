# Ablation Study — Attention Is All You Need

## Metodologia

Eksperymenty używają małego modelu ablacyjnego (d=256, 3 warstwy, 4 głowy), trenowanego na **2000 zdaniach** przez **10 epok**. Dzięki temu każdy eksperyment zajmuje 30–90 min zamiast dni.

Ewaluacja: sacrebleu na **500 zdaniach** z `newstest2014` przy beam_size=4.

Baseline i każdy eksperyment konfigurowany są przez Hydra — szczegóły w `configs/ablations/abl_*.yaml`.

---

## Wyniki

| Eksperyment | Config | Opis | BLEU | Delta |
|---|---|---|---:|---:|
| Baseline | `ablations/abl_base` | sinusoidal PE, 4 heads, label_smoothing=0.1 | pending | — |
| No Positional Encoding | `ablations/abl_no_pos_enc` | `positional_encoding: none` | pending | pending |
| Single Head Attention | `ablations/abl_single_head` | `h: 1`, 12 epok | pending | pending |
| Learned PE | `ablations/abl_learned_pe` | `positional_encoding: learned` | pending | pending |
| No Label Smoothing | `ablations/abl_no_smoothing` | `label_smoothing: 0.0` | pending | pending |
| Checkpoint Averaging (Base) | — | averaged.pt vs step_039062.pt | pending | pending |

*Uzupełnij po uruchomieniu `notebooks/ablations.ipynb`.*

---

## Opis eksperymentów

### Baseline

Konfiguracja: `configs/ablations/abl_base.yaml`

Referencyjna ewaluacja małego modelu z domyślnymi hiperparametrami. Punkt odniesienia dla obliczania delt.

---

### Eksperyment 1 — No Positional Encoding

Konfiguracja: `configs/ablations/abl_no_pos_enc.yaml` → `positional_encoding: none`

Model widzi tokeny jako nieuporządkowany zbiór. Attention może skupiać się na treści, ale nie ma informacji o kolejności w zdaniu.

**Hipoteza:** duży spadek BLEU (~5–10 pkt) — tłumaczenie wymaga odwzorowania kolejności słów między językami.

---

### Eksperyment 2 — Single Head Attention

Konfiguracja: `configs/ablations/abl_single_head.yaml` → `h: 1`

Jeden head zamiast 4 — model nie może równolegle śledzić różnych typów zależności (podmiot, orzeczenie, dopełnienie, czas).

**Hipoteza:** spadek BLEU ~2–5 pkt, wyraźny przy złożonych zdaniach.

---

### Eksperyment 3 — Learned Positional Encoding

Konfiguracja: `configs/ablations/abl_learned_pe.yaml` → `positional_encoding: learned`

Pozycje uczone przez gradient descent (nn.Embedding) zamiast deterministycznych sinusoid.

**Hipoteza:** wynik podobny do baseline (±1 BLEU). Sinusoidal PE ekstrapoluje na dłuższe sekwencje; learned PE nie. Przy krótkim treningu na małym zbiorze różnica powinna być minimalna.

---

### Eksperyment 4 — No Label Smoothing

Konfiguracja: `configs/ablations/abl_no_smoothing.yaml` → `label_smoothing: 0.0`

Model optymalizuje hard one-hot targets. Loss spada szybciej, ale model staje się overconfident w top-1 tokenach i słabiej generalizuje.

**Hipoteza:** niższy BLEU mimo niższego loss treningowego.

---

### Eksperyment 5 — Checkpoint Averaging

Brak treningu — ewaluacja istniejących checkpointów `checkpoints/base_en_de/`.

Porównuje `averaged.pt` (uśrednienie ostatnich 5 checkpointów) z pojedynczym `step_039062.pt`.

**Hipoteza:** averaged daje +0.5–2 BLEU przez redukcję szumu w wagach i lepszą generalizację.

---

## Jak uruchomić

```bash
# Prerequisite: dane i tokenizer muszą być gotowe
make download-data
make train-tokenizer

# Otwórz notebook
jupyter lab notebooks/ablations.ipynb

# Lub uruchom pojedynczy eksperyment z CLI
python train.py --config-name ablations/abl_no_pos_enc

# Szybki smoke-test (50 kroków)
python train.py --config-name ablations/abl_base training.max_steps=50 mlflow.enabled=false
```
