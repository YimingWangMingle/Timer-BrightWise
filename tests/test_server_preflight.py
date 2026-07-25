from tsfm.preflight import collect_environment_report, read_cgroup_memory_limit


def test_cgroup_v2_limit_parser_handles_bytes_and_max(tmp_path) -> None:
    path = tmp_path / "memory.max"
    path.write_text("96636764160\n", encoding="ascii")
    assert read_cgroup_memory_limit(path) == 90 * 1024**3

    path.write_text("max\n", encoding="ascii")
    assert read_cgroup_memory_limit(path) is None


def test_environment_report_contains_required_keys(tmp_path) -> None:
    report = collect_environment_report(tmp_path)

    assert {
        "python",
        "torch",
        "cuda_runtime",
        "driver",
        "gpu",
        "bf16_supported",
        "cgroup_memory_bytes",
        "disk",
    } <= report.keys()
    assert report["disk"]["path"] == str(tmp_path.resolve())
