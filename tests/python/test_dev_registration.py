from types import SimpleNamespace

from scripts import litepgo_dev


def test_windows_source_copies_refresh_and_remove_deleted_files(tmp_path, monkeypatch):
    source = tmp_path / "source"
    package = tmp_path / "build" / "litepgo"
    source.mkdir()
    (source / "__init__.py").write_text("value = 1\n")
    (source / "removed.py").write_text("old = True\n")
    monkeypatch.setattr(litepgo_dev, "os", SimpleNamespace(name="nt"))
    litepgo_dev.link_sources(source, package)
    assert (package / "__init__.py").read_text() == "value = 1\n"
    assert not (package / "__init__.py").is_symlink()
    (source / "__init__.py").write_text("value = 2\n")
    (source / "removed.py").unlink()
    litepgo_dev.link_sources(source, package)
    assert (package / "__init__.py").read_text() == "value = 2\n"
    assert not (package / "removed.py").exists()


def test_import_check_accepts_build_paths_with_spaces(tmp_path, monkeypatch):
    package = tmp_path / "build with spaces" / "litepgo"
    output = f"{package / '__init__.py'}\n{package / '_core.pyd'}\n"
    monkeypatch.setattr(litepgo_dev.subprocess, "run", lambda *a, **k: SimpleNamespace(
        returncode=0, stdout=output, stderr=""))
    litepgo_dev.check_import(package)
