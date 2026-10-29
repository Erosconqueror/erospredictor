"""Isolation for UI tests: never read or write the developer's real preferences."""

from unittest.mock import patch


def isolate_preferences(test):
    for name, value in (("load_ui_preferences", {}), ("save_ui_preferences", True)):
        mocked = patch("view.main_window." + name, return_value=value)
        mocked.start()
        test.addCleanup(mocked.stop)
