import torch

from tsfm.data import SyntheticTimeSeriesDataset, normalize_context_target


def test_synthetic_dataset_is_deterministic_per_index() -> None:
    dataset = SyntheticTimeSeriesDataset(num_samples=8, total_length=80, seed=123)
    recreated = SyntheticTimeSeriesDataset(num_samples=8, total_length=80, seed=123)
    torch.testing.assert_close(dataset[3], dataset[3], rtol=0.0, atol=0.0)
    torch.testing.assert_close(dataset[3], recreated[3], rtol=0.0, atol=0.0)
    assert not torch.equal(dataset[3], dataset[4])


def test_synthetic_dataset_has_requested_shape_and_finite_values() -> None:
    dataset = SyntheticTimeSeriesDataset(num_samples=5, total_length=48, seed=9)
    assert len(dataset) == 5
    assert dataset[0].shape == (48,)
    assert dataset[0].dtype == torch.float32
    assert torch.isfinite(dataset[0]).all()


def test_normalization_uses_context_statistics_for_both_tensors() -> None:
    context = torch.tensor([[1.0, 3.0, 5.0], [10.0, 10.0, 10.0]])
    target = torch.tensor([[7.0, 9.0], [12.0, 8.0]])
    batch = normalize_context_target(context, target)
    expected_mean = context.mean(dim=-1, keepdim=True)
    expected_scale = torch.sqrt(
        context.var(dim=-1, unbiased=False, keepdim=True) + 1e-5
    )
    torch.testing.assert_close(batch.mean, expected_mean)
    torch.testing.assert_close(batch.scale, expected_scale)
    torch.testing.assert_close(batch.context, (context - expected_mean) / expected_scale)
    torch.testing.assert_close(batch.target, (target - expected_mean) / expected_scale)
