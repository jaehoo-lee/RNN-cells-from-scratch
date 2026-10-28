# RNN-cells-from-scratch

Elman RNN, LSTM, and GRU cells implemented in PyTorch without `torch.nn` recurrent modules, with tests that check them against `torch.nn`.

The cells were written for an ongoing research project on time-series classification. This repository contains only the cell implementations and their tests; no research data, features, or results are included.

## Files

| File | Contents |
|---|---|
| `model.py` | `Elman_RNNCell`, `LSTM_Cell`, `GRU_Cell`, and sequence models (`Elman_RNN`, `LSTM`, `GRU`) that unroll a cell over time and apply a linear head to the last hidden state |
| `test_model.py` | Equivalence and sanity tests |

Weights are stored as `(in_features, out_features)` and applied as `x @ W` (row-vector convention). `torch.nn` stores them as `(out_features, in_features)`, so the tests transpose them before comparing.

## Cells

`@` is matrix multiplication, `*` is element-wise product.

**Elman RNN**

```
h_t = tanh(x_t @ W_xh + h_{t-1} @ W_hh + b_h)
```

**LSTM**

```
f_t = sigmoid(x_t @ W_xf + h_{t-1} @ W_hf + b_f)
i_t = sigmoid(x_t @ W_xi + h_{t-1} @ W_hi + b_i)
o_t = sigmoid(x_t @ W_xo + h_{t-1} @ W_ho + b_o)
c_candidates = tanh(x_t @ W_xc + h_{t-1} @ W_hc + b_c)
c_t = f_t * c_{t-1} + i_t * c_candidates
h_t = o_t * tanh(c_t)
```

**GRU**

```
z_t = sigmoid(x_t @ W_xz + h_{t-1} @ W_hz + b_z)
r_t = sigmoid(x_t @ W_xr + h_{t-1} @ W_hr + b_r)
h_tilde = tanh(x_t @ W_xh + (r_t * h_{t-1}) @ W_hh + b_h)
h_t = z_t * h_tilde + (1 - z_t) * h_{t-1}
```

### How this GRU differs from `torch.nn.GRUCell`

1. **Reset gate placement.** Here the reset gate is applied to `h_{t-1}` *before* the matrix multiplication, as in Cho et al. (2014). `torch.nn.GRUCell` applies it *after*: `r_t * (h_{t-1} @ W_hh + b_hn)`. The two give different outputs in general.
2. **Update gate convention.** Here `z_t` weights the candidate; in Cho et al. and `torch.nn`, `z_t` weights `h_{t-1}`. This is only a relabeling: since `1 - sigmoid(a) = sigmoid(-a)`, negating the update-gate weights converts one into the other.

## Verification

All comparisons load the same weights into both implementations and run in float64.

| Test | What it checks |
|---|---|
| `test_elman_matches_torch` | Output equals `torch.nn.RNNCell` |
| `test_lstm_matches_torch` | `h_t` and `c_t` equal `torch.nn.LSTMCell` (gates stacked in its order: i, f, g, o) |
| `test_gru_matches_torch_when_reset_gate_is_one` | With `b_r = 100`, `r_t` is exactly 1.0, so reset placement no longer matters; the output then equals `torch.nn.GRUCell` with negated update-gate weights. This covers the update gate, `W_xh`, `W_hh`, and the final interpolation. |
| `test_gru_differs_from_torch_in_general` | With an ordinary reset gate, outputs differ, documenting that the difference is intentional |
| `test_models_run_and_backprop` | Sequence models return shape `(batch, num_classes)` and every parameter receives a gradient |

Limitation: because `r_t` is fixed to 1 in the GRU equivalence test, the reset-gate weights (`W_xr`, `W_hr`, `b_r`) are not covered by an equivalence check.

## Usage

```python
import torch
from model import LSTM

model = LSTM(input_size=3, hidden_size=5)
X = torch.randn(4, 7, 3)  # (batch, sequence_length, features)
out = model(X)            # (4, 1)
```

## Running the tests

```bash
pip install torch pytest
pytest test_model.py -v
```

## Reference

K. Cho et al., "Learning Phrase Representations using RNN Encoder–Decoder for Statistical Machine Translation," EMNLP 2014.