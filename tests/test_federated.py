import pandas as pd
from asfl_iot.federated import run_federated


def test_run_federated_smoke():
    # Fast 2-round test with 3 clients
    df = run_federated(
        dataset_name="digits",
        method="test_asfl",
        seed=42,
        rounds=2,
        num_clients=3,
        use_datt=True,
        use_async=True,
        use_brae=True,
        dp_mode="sadp",
        use_pfl=True,
    )
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2
    assert "accuracy" in df.columns
    assert "macro_f1" in df.columns
    assert not df["accuracy"].isna().any()
