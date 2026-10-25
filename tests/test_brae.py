import torch
from asfl_iot.brae import brae_filter, make_byzantine, cosine_similarity
from asfl_iot.models import SmallMLP, get_params


def test_brae_filter_detects_sign_flip():
    global_model = SmallMLP(input_dim=10, num_classes=2, hidden=16)
    global_params = get_params(global_model)

    benign_params = []
    for _ in range(7):
        m = SmallMLP(input_dim=10, num_classes=2, hidden=16)
        # Small perturbation around global model
        p = {k: v + torch.randn_like(v) * 0.01 for k, v in global_params.items()}
        benign_params.append(p)

    malicious_params = []
    for _ in range(3):
        m = SmallMLP(input_dim=10, num_classes=2, hidden=16)
        p = make_byzantine(get_params(m), attack="sign_flip")
        malicious_params.append(p)

    all_params = benign_params + malicious_params
    keep, info = brae_filter(global_params, all_params)

    assert info["detected"] >= 2
    # Ensure benign clients are mostly retained
    for i in range(7):
        assert i in keep
