from china_ir.paths import DATA_RAW, DATA_WORK, FIGURES, OUTPUTS, REPO_ROOT, ensure_dirs


def test_paths_exist_and_writable():
    assert REPO_ROOT.exists()
    ensure_dirs()
    for p in (DATA_RAW, DATA_WORK, FIGURES, OUTPUTS):
        assert p.exists()
        assert p.is_dir()
