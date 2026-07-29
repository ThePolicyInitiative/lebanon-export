from pathlib import Path

APP = Path(__file__).resolve().parents[1] / "app.py"
TEXT = APP.read_text(encoding="utf-8")

def test_fixed_marker_columns_exist():
    assert "grid-template-columns: 1.72rem minmax(0, 1fr)" in TEXT
    assert "counter-reset: agent-list-item" in TEXT
    assert 'content: counter(agent-list-item) "."' in TEXT

def test_bullet_column_exists():
    assert "grid-template-columns: 0.78rem minmax(0, 1fr)" in TEXT
    assert 'content: "-"' in TEXT

def test_wrapped_content_uses_second_column():
    assert ".stChatMessage ol > li > *" in TEXT
    assert "grid-column: 2 !important" in TEXT
