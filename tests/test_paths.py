from china_ir import paths as P


def test_paths_exist_and_writable():
    P.ensure_dirs()
    assert P.REPO_ROOT.exists()
    for p in (P.DATA_RAW, P.DATA_WORK, P.FIGURES, P.OUTPUTS):
        assert p.exists()
        assert p.is_dir()
