import torch
from asfl_iot.models import SmallMLP, get_params, set_params, flatten_params, average_params


def test_small_mlp_forward():
    model = SmallMLP(input_dim=64, num_classes=10, hidden=96)
    x = torch.randn(4, 64)
    out = model(x)
    assert out.shape == (4, 10)


def test_parameter_manipulation():
    model = SmallMLP(input_dim=30, num_classes=4, hidden=64)
    params = get_params(model)
    assert "body.0.weight" in params
    assert "head.weight" in params

    flat = flatten_params(params)
    assert flat.ndim == 1
    assert flat.numel() > 0

    flat_body = flatten_params(params, body_only=True)
    assert flat_body.numel() < flat.numel()

    # Test averaging
    model2 = SmallMLP(input_dim=30, num_classes=4, hidden=64)
    params2 = get_params(model2)
    avg = average_params([params, params2], weights=[0.5, 0.5])
    assert "body.0.weight" in avg
    set_params(model, avg)
