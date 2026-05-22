from coding_agent.memory.stores.file_summary import detect_language, file_hash, is_summary_valid, summarize_file


def test_python_summary_extracts_imports_classes_methods_and_docstrings(tmp_path):
    path = tmp_path / "module.py"
    path.write_text(
        'import json\nfrom pathlib import Path\n\nclass Worker:\n    """Does work."""\n    def run(self):\n        return Path()\n\nasync def main():\n    pass\n',
        encoding="utf-8",
    )

    summary = summarize_file(path, tmp_path)

    assert summary.language == "python"
    assert summary.imports == ["import json", "from pathlib import Path"]
    names = [symbol.name for symbol in summary.symbols]
    assert "Worker" in names
    assert "Worker.run" in names
    assert "main" in names
    assert summary.symbols[0].docstring == "Does work."
    assert summary.symbols[0].preview[0] == "class Worker:"


def test_javascript_summary_extracts_imports_and_symbols(tmp_path):
    path = tmp_path / "app.ts"
    path.write_text(
        'import x from "x";\nexport class App {}\nexport const run = () => true;\nfunction helper() {}\n',
        encoding="utf-8",
    )

    summary = summarize_file(path, tmp_path)

    assert summary.language == "typescript"
    assert summary.imports == ['import x from "x";']
    assert [symbol.name for symbol in summary.symbols] == ["App", "run", "helper"]


def test_summary_validity_uses_hash_and_stale_flag(tmp_path):
    path = tmp_path / "notes.md"
    path.write_text("# Title\n", encoding="utf-8")
    summary = summarize_file(path, tmp_path).to_dict()

    assert detect_language(path) == "markdown"
    assert is_summary_valid(summary, path) is True
    path.write_text("# Changed\n", encoding="utf-8")
    assert is_summary_valid(summary, path) is False
    summary["content_hash"] = file_hash(path)
    summary["stale"] = True
    assert is_summary_valid(summary, path) is False
