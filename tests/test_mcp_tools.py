from unittest.mock import MagicMock, patch

import think.mcp_tools as mcp_tools


def test_todo_list_success_returns_numbered_markdown():
    mock_checklist = MagicMock()
    mock_checklist.numbered.return_value = "1: - [ ] Investigate"

    with patch.object(
        mcp_tools.todo.TodoChecklist,
        "load",
        return_value=mock_checklist,
    ) as load_mock:
        result = mcp_tools.todo_list("20240101")

    load_mock.assert_called_once_with("20240101")
    assert result == {"day": "20240101", "markdown": "1: - [ ] Investigate"}
    mock_checklist.numbered.assert_called_once_with()
