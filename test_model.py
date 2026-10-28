import torch
import torch.nn as nn

from model import Elman_RNNCell, LSTM_Cell, GRU_Cell, Elman_RNN, LSTM, GRU

torch.manual_seed(0)
B, I, H, T = 4, 3, 5, 7  # batch, input size, hidden size, sequence length


def random_inputs():
    x = torch.randn(B, I, dtype=torch.float64)
    h = torch.randn(B, H, dtype=torch.float64)
    c = torch.randn(B, H, dtype=torch.float64)
    return x, h, c


def test_elman_matches_torch():
    cell = Elman_RNNCell(I, H).double()
    ref = nn.RNNCell(I, H).double()
    with torch.no_grad():
        # torch.nn stores weights as (out, in), ours as (in, out)
        ref.weight_ih.copy_(cell.W_xh.T)
        ref.weight_hh.copy_(cell.W_hh.T)
        ref.bias_ih.copy_(cell.b_h)
        ref.bias_hh.zero_()  # torch.nn has a second bias; ours has one

    x, h, _ = random_inputs()
    torch.testing.assert_close(cell(x, h), ref(x, h))


def test_lstm_matches_torch():
    cell = LSTM_Cell(I, H).double()
    ref = nn.LSTMCell(I, H).double()
    with torch.no_grad():
        # torch.nn.LSTMCell stacks gates in the order (i, f, g, o)
        ref.weight_ih.copy_(torch.cat([cell.W_xi, cell.W_xf, cell.W_xc, cell.W_xo], dim=1).T)
        ref.weight_hh.copy_(torch.cat([cell.W_hi, cell.W_hf, cell.W_hc, cell.W_ho], dim=1).T)
        ref.bias_ih.copy_(torch.cat([cell.b_i, cell.b_f, cell.b_c, cell.b_o]))
        ref.bias_hh.zero_()

    x, h, c = random_inputs()
    h_ours, c_ours = cell(x, h, c)
    h_ref, c_ref = ref(x, (h, c))
    torch.testing.assert_close(h_ours, h_ref)
    torch.testing.assert_close(c_ours, c_ref)


def gru_reference(cell):
    """torch.nn.GRUCell loaded with our GRU weights.

    torch.nn.GRUCell stacks gates in the order (r, z, n) and computes
    h_t = (1 - z) * n + z * h_{t-1}, so its z equals 1 - (our z_t).
    Since 1 - sigmoid(a) = sigmoid(-a), the update-gate weights are negated.
    """
    ref = nn.GRUCell(I, H).double()
    with torch.no_grad():
        ref.weight_ih.copy_(torch.cat([cell.W_xr, -cell.W_xz, cell.W_xh], dim=1).T)
        ref.weight_hh.copy_(torch.cat([cell.W_hr, -cell.W_hz, cell.W_hh], dim=1).T)
        ref.bias_ih.copy_(torch.cat([cell.b_r, -cell.b_z, cell.b_h]))
        ref.bias_hh.zero_()
    return ref


def test_gru_matches_torch_when_reset_gate_is_one():
    # With r_t = 1 the reset-gate placement no longer matters,
    # so the two formulations must agree exactly.
    cell = GRU_Cell(I, H).double()
    with torch.no_grad():
        cell.b_r.fill_(100.0)  # sigmoid(100) == 1.0 in float64
    ref = gru_reference(cell)

    x, h, _ = random_inputs()
    torch.testing.assert_close(cell(x, h), ref(x, h))


def test_gru_differs_from_torch_in_general():
    # Ours: (r_t * h_{t-1}) @ W_hh / torch.nn: r_t * (h_{t-1} @ W_hh)
    cell = GRU_Cell(I, H).double()
    ref = gru_reference(cell)

    x, h, _ = random_inputs()
    assert not torch.allclose(cell(x, h), ref(x, h))


def test_models_run_and_backprop():
    X = torch.randn(B, T, I)
    for model in [Elman_RNN(I, H), LSTM(I, H), GRU(I, H)]:
        out = model(X)
        assert out.shape == (B, 1)

        out.sum().backward()
        for name, p in model.named_parameters():
            assert p.grad is not None, f"{type(model).__name__}.{name} got no gradient"


if __name__ == "__main__":
    tests = [
        test_elman_matches_torch,
        test_lstm_matches_torch,
        test_gru_matches_torch_when_reset_gate_is_one,
        test_gru_differs_from_torch_in_general,
        test_models_run_and_backprop,
    ]
    for test in tests:
        test()
        print("passed:", test.__name__)