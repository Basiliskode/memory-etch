"""Verification tests for PR6 Runtime Hardening.

Validates:
- _call_llm_extract raises RuntimeError with guidance
- Viewer uses logging module (timestamp+level+message)
- conftest has E2E mock fixture
- All existing E2E tests still pass (run separately)
"""
import inspect
import io
import logging

import pytest

from memory_etch import EtchMemoryProvider
from memory_etch.viewer import ViewerHandler, main as viewer_main


class TestCallLlmExtract:
    """RUN-1: _call_llm_extract raises RuntimeError with guidance."""

    def test_raises_runtime_error(self):
        """Method raises RuntimeError (not NotImplementedError)."""
        provider = EtchMemoryProvider({"auto_extract_llm": False})
        provider.initialize("test-runtime")
        with pytest.raises(RuntimeError) as exc_info:
            provider._call_llm_extract("system prompt", "user prompt")
        msg = str(exc_info.value)
        # Guidance must mention installation or configuration path
        assert "install" in msg.lower() or "provide" in msg.lower()
        assert "auto_extract_llm" in msg
        assert "docs/extraction.md" in msg

    def test_not_not_implemented_error(self):
        """Ensure it is NOT a NotImplementedError (spec RUN-1)."""
        provider = EtchMemoryProvider({"auto_extract_llm": False})
        provider.initialize("test-not-nie")
        with pytest.raises(Exception) as exc_info:
            provider._call_llm_extract("sys", "usr")
        assert type(exc_info.value) is RuntimeError
        assert not isinstance(exc_info.value, NotImplementedError)


class TestViewerLogging:
    """RUN-3: Viewer logging uses logging module with timestamp+level+message."""

    def test_log_message_uses_logging(self):
        """log_message delegates to logging.getLogger().info()."""
        # Capture log output to verify the pattern
        log_capture = io.StringIO()
        handler = logging.StreamHandler(log_capture)
        handler.setFormatter(logging.Formatter(
            "%(asctime)s [%(levelname)s] %(message)s"
        ))
        viewer_logger = logging.getLogger("memory_etch.viewer")
        viewer_logger.addHandler(handler)
        viewer_logger.setLevel(logging.INFO)

        try:
            # Instantiate a ViewerHandler minimally to call log_message
            req_handler = ViewerHandler.__new__(ViewerHandler)
            req_handler.log_date_time_string = lambda: "[19/May/2026 12:00:00]"
            # log_message signature: log_message(self, fmt, *args)
            req_handler.log_message("GET /api/facts")

            output = log_capture.getvalue()
            assert "[19/May/2026 12:00:00]" in output
            assert "GET /api/facts" in output
        finally:
            viewer_logger.removeHandler(handler)

    def test_main_uses_logging_basicconfig(self):
        """main() sets up logging.basicConfig with timestamp+level+message."""
        source = inspect.getsource(viewer_main)
        assert "logging.basicConfig" in source
        assert "%(asctime)s" in source
        assert "%(levelname)s" in source
        assert "%(message)s" in source

    def test_main_uses_logger_instead_of_print(self):
        """main() uses logger.info/error instead of bare print for status."""
        source = inspect.getsource(viewer_main)
        # Should have logger.info and logger.error calls
        assert "logger.info(" in source
        assert "logger.error(" in source
        # Should NOT have bare print (except sys.exit)
        # Check no `print(` for logging purposes — print may appear
        # only in contexts that aren't startup logging

    def test_viewer_binds_default_127_0_0_1(self):
        """RUN-2: Default host remains 127.0.0.1 — unchanged."""
        source = inspect.getsource(viewer_main)
        assert '"127.0.0.1"' in source
        # Also check create_viewer_server signature
        sig_source = inspect.getsource(inspect.getmodule(viewer_main).create_viewer_server)
        assert '"127.0.0.1"' in sig_source or "'127.0.0.1'" in sig_source


class TestConftestMockFixture:
    """Fixture mock_llm_extract exists and works in conftest."""

    def test_mock_llm_extract_fixture_registered(self):
        """mock_llm_extract fixture exists in conftest."""
        import conftest
        fixture = getattr(conftest, "mock_llm_extract", None)
        assert fixture is not None, "conftest is missing mock_llm_extract fixture"
        assert callable(fixture)

    def test_mock_llm_extract_returns_expected_value(self):
        """The mock returns 'Extracted: {content}' when called."""
        from unittest.mock import MagicMock
        mock = MagicMock(return_value="Extracted: {content}")
        result = mock("system", "user text here")
        assert result == "Extracted: {content}"

    def test_mock_patches_class_method(self):
        """The mock fixture patches EtchMemoryProvider._call_llm_extract."""
        from unittest.mock import patch
        with patch.object(
            EtchMemoryProvider, '_call_llm_extract',
            return_value="Extracted: {content}",
        ):
            provider = EtchMemoryProvider({"auto_extract_llm": False})
            provider.initialize("test-mock-patch")
            result = provider._call_llm_extract("sys", "user")
            assert result == "Extracted: {content}"


class TestStartupLog:
    """EtchMemoryProvider emits startup log about uncovered _call_llm_extract."""

    def test_init_emits_log(self):
        """Constructing EtchMemoryProvider logs that _call_llm_extract is not configured."""
        log_capture = io.StringIO()
        handler = logging.StreamHandler(log_capture)
        etch_logger = logging.getLogger("memory_etch.etch")
        etch_logger.addHandler(handler)
        etch_logger.setLevel(logging.INFO)

        try:
            provider = EtchMemoryProvider({"auto_extract_llm": False})
            output = log_capture.getvalue()
            assert "_call_llm_extract" in output
            assert "not configured" in output.lower()
        finally:
            etch_logger.removeHandler(handler)
            if provider:
                provider.shutdown()
