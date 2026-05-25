"""Tests for the interceptor subpackage — Phase 1: Foundation (Tasks 1.1-1.3)."""

import uuid
from unittest.mock import MagicMock, patch, ANY
from dataclasses import dataclass

import pytest

from memento import EtchStore
from memento.interceptor import InterceptorHandle, intercept, teardown_all


# ---------------------------------------------------------------------------
# Task 1.1: interceptor/__init__.py — intercept(), InterceptorHandle, teardown_all
# ---------------------------------------------------------------------------

class TestInterceptorHandle:
    """InterceptorHandle is a simple dataclass holding teardown state."""

    def test_handle_fields(self):
        handle = InterceptorHandle(
            name="test",
            original=object(),
            wrapped=object(),
            teardown=lambda: None,
        )
        assert handle.name == "test"
        assert callable(handle.teardown)


class TestInterceptUnknownTarget:
    """intercept() with unknown target names raises ValueError."""

    def test_unknown_target_raises(self, etch_store):
        with pytest.raises(ValueError, match="Unknown target"):
            intercept(etch_store, targets=["nonexistent_sdk"])

    def test_multiple_unknown_targets_raises(self, etch_store):
        with pytest.raises(ValueError, match="Unknown target"):
            intercept(etch_store, targets=["foo", "bar"])

    def test_mix_known_and_unknown_raises(self, etch_store):
        with pytest.raises(ValueError, match="Unknown target"):
            intercept(etch_store, targets=["openai", "bogus"])


class TestInterceptMissingSDK:
    """intercept() with a target whose SDK is not installed raises ImportError."""

    def test_openai_missing_raises(self, etch_store):
        with patch.dict("sys.modules", {"openai": None}):
            with pytest.raises(ImportError, match="openai"):
                intercept(etch_store, targets=["openai"])

    def test_anthropic_missing_raises(self, etch_store):
        with patch.dict("sys.modules", {"anthropic": None}):
            with pytest.raises(ImportError, match="anthropic"):
                intercept(etch_store, targets=["anthropic"])


class TestTeardownAll:
    """teardown_all() calls teardown on all handles."""

    def test_teardown_all_calls_each_handle(self):
        teardown_calls = []

        def make_handle(name):
            def td():
                teardown_calls.append(name)
            return InterceptorHandle(name=name, original=None, wrapped=None, teardown=td)

        handles = [make_handle("a"), make_handle("b"), make_handle("c")]
        teardown_all(handles)
        assert teardown_calls == ["a", "b", "c"]

    def test_teardown_all_empty_handles(self):
        """Calling teardown_all with an empty list does not raise."""
        teardown_all([])

    def test_teardown_all_idempotent(self):
        """Calling teardown_all twice is safe."""
        calls = []

        def td():
            calls.append("x")

        handles = [InterceptorHandle(name="x", original=None, wrapped=None, teardown=td)]
        teardown_all(handles)
        teardown_all(handles)
        assert calls == ["x", "x"]


class TestInterceptReturnsHandles:
    """intercept() returns a list of InterceptorHandle objects."""

    def test_intercept_returns_list(self, etch_store):
        """When targets=None, intercept() should return a list (possibly empty if no SDKs)."""
        handles = intercept(etch_store)
        assert isinstance(handles, list)
        # All items should be InterceptorHandle
        for h in handles:
            assert isinstance(h, InterceptorHandle)
        # Cleanup
        teardown_all(handles)

    def test_intercept_empty_targets_list_returns_empty(self, etch_store):
        """Explicit empty list returns empty handles list."""
        handles = intercept(etch_store, targets=[])
        assert handles == []

    def test_intercept_single_string_target(self, etch_store):
        """A single string target is accepted (normalized to list internally)."""
        with pytest.raises(ImportError):
            intercept(etch_store, targets="openai")


# ---------------------------------------------------------------------------
# Task 1.2: interceptor/generic.py — GenericInterceptor
# ---------------------------------------------------------------------------


class TestGenericInterceptorMetadata:
    """GenericInterceptor passes provenance metadata to add_fact."""

    def test_generic_forwards_source_harness(self):
        """GenericInterceptor passes source_harness='interceptor' and source_kind='conversation'."""
        store = MagicMock(spec=EtchStore)
        store.add_fact.return_value = 1

        def fake_llm(messages, **kwargs):
            return "Assistant response text"

        from memento.interceptor.generic import GenericInterceptor

        interceptor = GenericInterceptor(store, fake_llm)
        wrapped = interceptor.wrap()
        wrapped(messages=[{"role": "user", "content": "Hello"}])

        for call in store.add_fact.call_args_list:
            kwargs = call.kwargs
            assert kwargs.get("source_harness") == "interceptor"
            assert kwargs.get("source_kind") == "conversation"

    def test_generic_does_not_break_existing_behavior(self):
        """Adding provenance metadata does not break existing behavior."""
        store = MagicMock(spec=EtchStore)
        store.add_fact.return_value = 1

        def fake_llm(messages, **kwargs):
            return "Response"

        from memento.interceptor.generic import GenericInterceptor

        interceptor = GenericInterceptor(store, fake_llm)
        wrapped = interceptor.wrap()
        result = wrapped(messages=[{"role": "user", "content": "Hello"}])

        assert result == "Response"
        assert store.add_fact.call_count == 2

        # All other existing fields still present
        user_call = store.add_fact.call_args_list[0].kwargs
        assert user_call["category"] == "conversation"
        assert user_call["content"] == "user: Hello"
        assert "role:user" in user_call["tags"]
        assert user_call["trust_score"] == 0.9


class TestGenericInterceptorWrap:
    """GenericInterceptor.wrap() captures 2 facts per call."""

    def test_wrap_stores_two_facts(self):
        """wrap() returns a callable that stores user + assistant facts."""
        store = MagicMock(spec=EtchStore)
        store.add_fact.return_value = 1

        def fake_llm(messages, **kwargs):
            return "Assistant response text"

        from memento.interceptor.generic import GenericInterceptor

        interceptor = GenericInterceptor(store, fake_llm)
        wrapped = interceptor.wrap()

        result = wrapped(messages=[{"role": "user", "content": "Hello"}])

        assert result == "Assistant response text"
        # Must store exactly 2 facts
        assert store.add_fact.call_count == 2

    def test_fact_content_format(self):
        """Facts are stored with content='role: text' format."""
        store = MagicMock(spec=EtchStore)
        store.add_fact.return_value = 1

        def fake_llm(messages, **kwargs):
            return "Response text"

        from memento.interceptor.generic import GenericInterceptor

        interceptor = GenericInterceptor(store, fake_llm)
        wrapped = interceptor.wrap()
        wrapped(messages=[{"role": "user", "content": "Hello there"}])

        calls = store.add_fact.call_args_list
        # First call: user fact
        user_fact = calls[0].kwargs
        assert user_fact["content"] == "user: Hello there"
        assert user_fact["category"] == "conversation"
        assert "role:user" in user_fact["tags"]
        assert user_fact["trust_score"] == 0.9

        # Second call: assistant fact
        asst_fact = calls[1].kwargs
        assert asst_fact["content"] == "assistant: Response text"
        assert "role:assistant" in asst_fact["tags"]

    def test_conversation_id_applied(self):
        """A conversation_id is applied to all facts in the same session."""
        store = MagicMock(spec=EtchStore)
        store.add_fact.return_value = 1

        def fake_llm(messages, **kwargs):
            return "Response"

        from memento.interceptor.generic import GenericInterceptor

        interceptor = GenericInterceptor(store, fake_llm, conversation_id="my-test-id")
        wrapped = interceptor.wrap()
        wrapped(messages=[{"role": "user", "content": "Hi"}])

        for call in store.add_fact.call_args_list:
            assert "my-test-id" in call.kwargs.get("topic_key", "")

    def test_topic_key_format(self):
        """topic_key includes the conversation_id and turn info."""
        store = MagicMock(spec=EtchStore)
        store.add_fact.return_value = 1

        def fake_llm(messages, **kwargs):
            return "Response"

        from memento.interceptor.generic import GenericInterceptor

        interceptor = GenericInterceptor(store, fake_llm, conversation_id="conv-123")
        wrapped = interceptor.wrap()
        wrapped(messages=[{"role": "user", "content": "Hi"}])

        for call in store.add_fact.call_args_list:
            topic_key = call.kwargs["topic_key"]
            assert "conv-123" in topic_key
            assert topic_key.startswith("conversation/")

    def test_auto_conversation_id_generated(self):
        """When no conversation_id is given, one is auto-generated."""
        store = MagicMock(spec=EtchStore)
        store.add_fact.return_value = 1

        def fake_llm(messages, **kwargs):
            return "Response"

        from memento.interceptor.generic import GenericInterceptor

        interceptor = GenericInterceptor(store, fake_llm)
        wrapped = interceptor.wrap()
        wrapped(messages=[{"role": "user", "content": "Hi"}])

        for call in store.add_fact.call_args_list:
            topic_key = call.kwargs["topic_key"]
            assert topic_key.startswith("conversation/")
            # Verify the ID part is a valid UUID v4 (path: conversation/<uuid>/t0001/user)
            conv_id = topic_key.split("/")[1]
            uuid.UUID(conv_id, version=4)

    def test_wrapped_callable_signature(self):
        """The wrapped callable accepts messages and **kwargs, returns str."""
        store = MagicMock(spec=EtchStore)
        store.add_fact.return_value = 1

        def fake_llm(messages, model="default", **kwargs):
            return f"Response from {model}"

        from memento.interceptor.generic import GenericInterceptor

        interceptor = GenericInterceptor(store, fake_llm)
        wrapped = interceptor.wrap()

        result = wrapped(messages=[{"role": "user", "content": "Test"}], model="gpt-4")
        assert result == "Response from gpt-4"

    def test_extra_kwargs_passed_to_original(self):
        """Extra kwargs are forwarded to the original function."""
        store = MagicMock(spec=EtchStore)
        store.add_fact.return_value = 1

        def fake_llm(messages, **kwargs):
            assert kwargs.get("temperature") == 0.7
            return "ok"

        from memento.interceptor.generic import GenericInterceptor

        interceptor = GenericInterceptor(store, fake_llm)
        wrapped = interceptor.wrap()
        wrapped(messages=[{"role": "user", "content": "Test"}], temperature=0.7)


class TestGenericInterceptorContextManager:
    """GenericInterceptor works as a context manager."""

    def test_context_manager_returns_wrapped(self):
        store = MagicMock(spec=EtchStore)
        store.add_fact.return_value = 1

        def fake_llm(messages, **kwargs):
            return "ok"

        from memento.interceptor.generic import GenericInterceptor

        with GenericInterceptor(store, fake_llm) as wrapped:
            result = wrapped(messages=[{"role": "user", "content": "Test"}])
            assert result == "ok"
        assert store.add_fact.call_count == 2

    def test_teardown_restores_original(self):
        """After teardown(), calling the original fn directly no longer stores facts."""
        store = MagicMock(spec=EtchStore)
        store.add_fact.return_value = 1

        original_fn = lambda messages, **kwargs: "original"
        original_id = id(original_fn)

        from memento.interceptor.generic import GenericInterceptor

        interceptor = GenericInterceptor(store, original_fn)
        wrapped = interceptor.wrap()

        # Wrapped function stores facts
        wrapped(messages=[{"role": "user", "content": "A"}])
        assert store.add_fact.call_count == 2

        # Teardown restores
        interceptor.teardown()

        # After teardown, wrapped function still calls original but no facts stored
        store.add_fact.reset_mock()
        wrapped(messages=[{"role": "user", "content": "B"}])
        assert store.add_fact.call_count == 0


# ---------------------------------------------------------------------------
# Task 1.3: interceptor/extract.py — extract_conversation
# ---------------------------------------------------------------------------


class TestExtractConversation:
    """extract_conversation runs LLM extraction on conversation turns."""

    def test_extract_returns_count(self):
        """extract_conversation returns the number of extracted facts."""
        store = MagicMock(spec=EtchStore)
        store.add_fact.return_value = 42

        conversation = [
            {"content": "user: Hello", "role": "user"},
            {"content": "assistant: Hi there!", "role": "assistant"},
        ]

        def llm_extract_fn(text):
            return ["User likes Python", "User uses VS Code"]

        from memento.interceptor.extract import extract_conversation

        count = extract_conversation(conversation, store, llm_extract_fn)
        assert count == 2

    def test_extracted_facts_have_correct_category(self):
        """Extracted facts are stored with category='extracted'."""
        store = MagicMock(spec=EtchStore)
        store.add_fact.return_value = 1

        conversation = [
            {"content": "user: Hello", "role": "user"},
        ]

        def llm_extract_fn(text):
            return ["User likes Python"]

        from memento.interceptor.extract import extract_conversation

        extract_conversation(conversation, store, llm_extract_fn)

        fact_call = store.add_fact.call_args
        assert fact_call.kwargs["category"] == "extracted"
        assert "interceptor" in fact_call.kwargs["tags"]

    def test_extract_passes_conversation_text(self):
        """The LLM extract function receives full conversation text."""
        store = MagicMock(spec=EtchStore)
        store.add_fact.return_value = 1

        conversation = [
            {"content": "user: Hello", "role": "user"},
            {"content": "assistant: Hi there!", "role": "assistant"},
        ]

        received_text = None

        def llm_extract_fn(text):
            nonlocal received_text
            received_text = text
            return ["User likes Python"]

        from memento.interceptor.extract import extract_conversation

        extract_conversation(conversation, store, llm_extract_fn)
        assert received_text is not None
        assert "user: Hello" in received_text
        assert "assistant: Hi there!" in received_text

    def test_empty_extract_returns_zero(self):
        """When extract function returns empty list, count is 0."""
        store = MagicMock(spec=EtchStore)
        store.add_fact.return_value = 1

        conversation = [
            {"content": "user: Hi", "role": "user"},
        ]

        def llm_extract_fn(text):
            return []

        from memento.interceptor.extract import extract_conversation

        count = extract_conversation(conversation, store, llm_extract_fn)
        assert count == 0
        store.add_fact.assert_not_called()

    def test_extract_empty_conversation(self):
        """Empty conversation produces empty text and returns 0."""
        store = MagicMock(spec=EtchStore)
        store.add_fact.return_value = 1

        def llm_extract_fn(text):
            return []

        from memento.interceptor.extract import extract_conversation

        count = extract_conversation([], store, llm_extract_fn)
        assert count == 0


# ---------------------------------------------------------------------------
# Phase 2: SDK Wrappers
# ---------------------------------------------------------------------------
# Task 2.1: interceptor/openai.py
# ---------------------------------------------------------------------------


def _make_mock_chat_completion(text="Mock response"):
    """Create a minimal ChatCompletion-like mock object."""
    choice = MagicMock()
    choice.message.content = text
    choice.message.role = "assistant"
    choice.finish_reason = "stop"
    choice.index = 0

    usage = MagicMock()
    usage.prompt_tokens = 10
    usage.completion_tokens = 5
    usage.total_tokens = 15

    completion = MagicMock()
    completion.id = "chatcmpl-123"
    completion.object = "chat.completion"
    completion.created = 1700000000
    completion.model = "gpt-4"
    completion.choices = [choice]
    completion.usage = usage

    return completion


def _make_fake_openai_module():
    """Create a fake 'openai' module with the required structure for interception."""
    import types

    # We need: openai.OpenAI.chat.completions.create
    fake_create = MagicMock(name="original_openai_create")

    fake_completions = MagicMock(name="completions", spec=[])
    fake_completions.create = fake_create

    fake_chat = MagicMock(name="chat", spec=[])
    fake_chat.completions = fake_completions

    fake_openai_class = MagicMock(name="OpenAI", spec=[])
    fake_openai_class.chat = fake_chat
    # Note: the real openai.OpenAI.chat is an instance attribute set in __init__,
    # but the monkey-patch needs to reach it via the class or instance path.
    # We'll make it accessible both ways.

    fake_openai_module = types.ModuleType("openai")
    fake_openai_module.OpenAI = fake_openai_class
    fake_openai_module.__version__ = "1.0.0"

    return fake_openai_module, fake_create


def _make_fake_anthropic_module():
    """Create a fake 'anthropic' module with the required structure for interception."""
    import types

    fake_create = MagicMock(name="original_anthropic_create")

    fake_messages = MagicMock(name="messages", spec=[])
    fake_messages.create = fake_create

    fake_anthropic_class = MagicMock(name="Anthropic", spec=[])
    fake_anthropic_class.messages = fake_messages

    fake_anthropic_module = types.ModuleType("anthropic")
    fake_anthropic_module.Anthropic = fake_anthropic_class
    fake_anthropic_module.__version__ = "0.30.0"

    return fake_anthropic_module, fake_create


class TestOpenAIWrapper:
    """OpenAI ChatCompletion wrapper captures facts."""

    OPENAI_MODULE, _ = _make_fake_openai_module()

    def test_openai_wrapper_stores_facts(self, etch_store):
        """Wrapped OpenAI call stores user + assistant facts.

        Calls through the patched class attribute directly (same path the
        interceptor patches) instead of instantiating a client, because
        the fake openai module uses MagicMock which creates independent
        mocks for each attribute chain.
        """
        completion = _make_mock_chat_completion("Response from GPT")

        fake_module = self._make_patched_openai()
        mock_create = fake_module.OpenAI.chat.completions.create
        mock_create.return_value = completion

        with patch.dict("sys.modules", {"openai": fake_module}):
            from memento.interceptor import intercept, teardown_all

            handles = intercept(etch_store, targets=["openai"])
            try:
                import openai
                result = openai.OpenAI.chat.completions.create(
                    model="gpt-4",
                    messages=[{"role": "user", "content": "Hello world"}],
                )
            finally:
                teardown_all(handles)

        assert result is completion
        assert result.choices[0].message.content == "Response from GPT"

        facts = etch_store.list_facts()
        contents = {f["content"] for f in facts}
        assert "user: Hello world" in contents
        assert "assistant: Response from GPT" in contents

    def test_openai_fact_tags_include_model(self, etch_store):
        """OpenAI facts include model name in tags."""
        completion = _make_mock_chat_completion("Response")

        fake_module = self._make_patched_openai()
        mock_create = fake_module.OpenAI.chat.completions.create
        mock_create.return_value = completion

        with patch.dict("sys.modules", {"openai": fake_module}):
            from memento.interceptor import intercept, teardown_all

            handles = intercept(etch_store, targets=["openai"])
            try:
                import openai
                openai.OpenAI.chat.completions.create(
                    model="gpt-4",
                    messages=[{"role": "user", "content": "Hi"}],
                )
            finally:
                teardown_all(handles)

        facts = etch_store.list_facts()
        for f in facts:
            assert "model:gpt-4" in f["tags"]

    def test_openai_streaming_raises(self, etch_store):
        """stream=True raises NotImplementedError."""
        fake_module = self._make_patched_openai()

        with patch.dict("sys.modules", {"openai": fake_module}):
            from memento.interceptor import intercept, teardown_all

            handles = intercept(etch_store, targets=["openai"])
            try:
                import openai
                with pytest.raises(NotImplementedError, match="[Ss]tream"):
                    openai.OpenAI.chat.completions.create(
                        model="gpt-4",
                        messages=[{"role": "user", "content": "Hi"}],
                        stream=True,
                    )
            finally:
                teardown_all(handles)

    def test_openai_teardown_restores_original(self):
        """After teardown, original function is restored (id comparison)."""
        fake_module, original_create = _make_fake_openai_module()
        original_id = id(original_create)

        with patch.dict("sys.modules", {"openai": fake_module}):
            from memento.interceptor import intercept, teardown_all

            store = MagicMock(spec=EtchStore)
            store.add_fact.return_value = 1

            handles = intercept(store, targets=["openai"])
            patched_id = id(fake_module.OpenAI.chat.completions.create)
            assert patched_id != original_id
            teardown_all(handles)

        # After teardown, restored to original
        restored_id = id(fake_module.OpenAI.chat.completions.create)
        assert restored_id == original_id

    def _make_patched_openai(self):
        """Return a fake openai module with a clean mock_create for each test."""
        fake_module, _ = _make_fake_openai_module()
        return fake_module


# ---------------------------------------------------------------------------
# Task 2.2: interceptor/anthropic.py — Anthropic Messages wrapper
# ---------------------------------------------------------------------------


def _make_mock_anthropic_message(text="Mock Anthropic response"):
    """Create a minimal Anthropic Message-like mock object."""
    content_block = MagicMock()
    content_block.type = "text"
    content_block.text = text

    message = MagicMock()
    message.id = "msg_123"
    message.model = "claude-3-5-sonnet"
    message.role = "assistant"
    message.content = [content_block]
    message.stop_reason = "end_turn"
    message.usage = MagicMock()
    message.usage.input_tokens = 10
    message.usage.output_tokens = 5

    return message


class TestAnthropicWrapper:
    """Anthropic Messages wrapper captures facts."""

    def test_anthropic_wrapper_stores_facts(self, etch_store):
        """Wrapped Anthropic call stores user + assistant facts."""
        message = _make_mock_anthropic_message("Response from Claude")

        fake_module = _make_fake_anthropic_module()[0]
        mock_create = fake_module.Anthropic.messages.create
        mock_create.return_value = message

        with patch.dict("sys.modules", {"anthropic": fake_module}):
            from memento.interceptor import intercept, teardown_all

            handles = intercept(etch_store, targets=["anthropic"])
            try:
                import anthropic
                result = anthropic.Anthropic.messages.create(
                    model="claude-3-5-sonnet",
                    messages=[{"role": "user", "content": "Hello Claude"}],
                )
            finally:
                teardown_all(handles)

        assert result is message
        assert result.content[0].text == "Response from Claude"

        facts = etch_store.list_facts()
        contents = {f["content"] for f in facts}
        assert "user: Hello Claude" in contents
        assert "assistant: Response from Claude" in contents

    def test_anthropic_fact_tags_include_model(self, etch_store):
        """Anthropic facts include model name in tags."""
        message = _make_mock_anthropic_message("Response")

        fake_module = _make_fake_anthropic_module()[0]
        mock_create = fake_module.Anthropic.messages.create
        mock_create.return_value = message

        with patch.dict("sys.modules", {"anthropic": fake_module}):
            from memento.interceptor import intercept, teardown_all

            handles = intercept(etch_store, targets=["anthropic"])
            try:
                import anthropic
                anthropic.Anthropic.messages.create(
                    model="claude-3-5-sonnet",
                    messages=[{"role": "user", "content": "Hi"}],
                )
            finally:
                teardown_all(handles)

        facts = etch_store.list_facts()
        for f in facts:
            assert "model:claude-3-5-sonnet" in f["tags"]

    def test_anthropic_streaming_raises(self, etch_store):
        """stream=True raises NotImplementedError."""
        fake_module = _make_fake_anthropic_module()[0]

        with patch.dict("sys.modules", {"anthropic": fake_module}):
            from memento.interceptor import intercept, teardown_all

            handles = intercept(etch_store, targets=["anthropic"])
            try:
                import anthropic
                with pytest.raises(NotImplementedError, match="[Ss]tream"):
                    anthropic.Anthropic.messages.create(
                        model="claude-3-5-sonnet",
                        messages=[{"role": "user", "content": "Hi"}],
                        stream=True,
                    )
            finally:
                teardown_all(handles)

    def test_anthropic_teardown_restores_original(self):
        """After teardown, original function is restored (id comparison)."""
        fake_module, original_create = _make_fake_anthropic_module()
        original_id = id(original_create)

        with patch.dict("sys.modules", {"anthropic": fake_module}):
            from memento.interceptor import intercept, teardown_all

            store = MagicMock(spec=EtchStore)
            store.add_fact.return_value = 1

            handles = intercept(store, targets=["anthropic"])
            patched_id = id(fake_module.Anthropic.messages.create)
            assert patched_id != original_id
            teardown_all(handles)

        restored_id = id(fake_module.Anthropic.messages.create)
        assert restored_id == original_id


class TestOpenAIMetadata:
    """OpenAI interceptor passes provenance metadata."""

    def test_openai_passes_source_harness_and_kind(self, etch_store):
        """OpenAI interceptor passes source_harness='openai' and source_kind='conversation'."""
        completion = _make_mock_chat_completion("Response from GPT")

        fake_module = _make_fake_openai_module()[0]
        mock_create = fake_module.OpenAI.chat.completions.create
        mock_create.return_value = completion

        with patch.dict("sys.modules", {"openai": fake_module}):
            from memento.interceptor import intercept, teardown_all

            handles = intercept(etch_store, targets=["openai"])
            try:
                import openai
                openai.OpenAI.chat.completions.create(
                    model="gpt-4",
                    messages=[{"role": "user", "content": "Hello"}],
                )
            finally:
                teardown_all(handles)

        facts = etch_store.list_facts()
        for f in facts:
            assert f["source_harness"] == "openai", f"Expected 'openai', got '{f['source_harness']}'"
            assert f["source_kind"] == "conversation", f"Expected 'conversation', got '{f['source_kind']}'"

    def test_openai_passes_model_as_source_agent(self, etch_store):
        """OpenAI interceptor passes model name as source_agent."""
        completion = _make_mock_chat_completion("Response from GPT")

        fake_module = _make_fake_openai_module()[0]
        mock_create = fake_module.OpenAI.chat.completions.create
        mock_create.return_value = completion

        with patch.dict("sys.modules", {"openai": fake_module}):
            from memento.interceptor import intercept, teardown_all

            handles = intercept(etch_store, targets=["openai"])
            try:
                import openai
                openai.OpenAI.chat.completions.create(
                    model="gpt-4-turbo",
                    messages=[{"role": "user", "content": "Hi"}],
                )
            finally:
                teardown_all(handles)

        facts = etch_store.list_facts()
        for f in facts:
            assert f["source_agent"] == "gpt-4-turbo", f"Expected 'gpt-4-turbo', got '{f['source_agent']}'"


class TestAnthropicMetadata:
    """Anthropic interceptor passes provenance metadata."""

    def test_anthropic_passes_source_harness_and_kind(self, etch_store):
        """Anthropic interceptor passes source_harness='anthropic' and source_kind='conversation'."""
        message = _make_mock_anthropic_message("Response from Claude")

        fake_module = _make_fake_anthropic_module()[0]
        mock_create = fake_module.Anthropic.messages.create
        mock_create.return_value = message

        with patch.dict("sys.modules", {"anthropic": fake_module}):
            from memento.interceptor import intercept, teardown_all

            handles = intercept(etch_store, targets=["anthropic"])
            try:
                import anthropic
                anthropic.Anthropic.messages.create(
                    model="claude-3-opus",
                    messages=[{"role": "user", "content": "Hello Claude"}],
                )
            finally:
                teardown_all(handles)

        facts = etch_store.list_facts()
        for f in facts:
            assert f["source_harness"] == "anthropic", f"Expected 'anthropic', got '{f['source_harness']}'"
            assert f["source_kind"] == "conversation", f"Expected 'conversation', got '{f['source_kind']}'"

    def test_anthropic_passes_model_as_source_agent(self, etch_store):
        """Anthropic interceptor passes model name as source_agent."""
        message = _make_mock_anthropic_message("Response")

        fake_module = _make_fake_anthropic_module()[0]
        mock_create = fake_module.Anthropic.messages.create
        mock_create.return_value = message

        with patch.dict("sys.modules", {"anthropic": fake_module}):
            from memento.interceptor import intercept, teardown_all

            handles = intercept(etch_store, targets=["anthropic"])
            try:
                import anthropic
                anthropic.Anthropic.messages.create(
                    model="claude-3-5-sonnet",
                    messages=[{"role": "user", "content": "Hi"}],
                )
            finally:
                teardown_all(handles)

        facts = etch_store.list_facts()
        for f in facts:
            assert f["source_agent"] == "claude-3-5-sonnet", f"Expected 'claude-3-5-sonnet', got '{f['source_agent']}'"


class TestExtractMetadata:
    """extract_conversation passes provenance metadata."""

    def test_extract_passes_source_harness_and_kind(self):
        """extract_conversation passes interceptor metadata to add_fact."""
        store = MagicMock(spec=EtchStore)
        store.add_fact.return_value = 1

        conversation = [
            {"content": "user: Hello", "role": "user"},
            {"content": "assistant: Hi there!", "role": "assistant"},
        ]

        def llm_extract_fn(text):
            return ["User likes Python"]

        from memento.interceptor.extract import extract_conversation

        extract_conversation(conversation, store, llm_extract_fn)

        call_kwargs = store.add_fact.call_args.kwargs
        assert call_kwargs.get("source_harness") == "interceptor"
        assert call_kwargs.get("source_kind") == "conversation"

    def test_extract_without_breaking_existing_fields(self):
        """Adding metadata preserves existing fields in extract."""
        store = MagicMock(spec=EtchStore)
        store.add_fact.return_value = 1

        conversation = [
            {"content": "user: Hello", "role": "user"},
        ]

        def llm_extract_fn(text):
            return ["New fact"]

        from memento.interceptor.extract import extract_conversation

        extract_conversation(conversation, store, llm_extract_fn)

        call_kwargs = store.add_fact.call_args.kwargs
        assert call_kwargs["category"] == "extracted"
        assert "interceptor" in call_kwargs["tags"]
        assert call_kwargs["trust_score"] == 0.9


# ---------------------------------------------------------------------------
# Phase 3: Integration
# ---------------------------------------------------------------------------
# Task 3.3: Additional integration & regression tests
# ---------------------------------------------------------------------------


class TestLazyImport:
    """Importing interceptor subpackage does NOT require SDKs installed."""

    def test_import_interceptor_without_sdks(self):
        """from memento.interceptor import intercept succeeds w/o SDKs."""
        from memento.interceptor import intercept, teardown_all, InterceptorHandle
        assert callable(intercept)
        assert callable(teardown_all)

    def test_import_generic_without_sdks(self):
        """GenericInterceptor is importable without SDKs."""
        from memento.interceptor.generic import GenericInterceptor
        assert callable(GenericInterceptor)

    def test_import_extract_without_sdks(self):
        """extract_conversation is importable without SDKs."""
        from memento.interceptor.extract import extract_conversation
        assert callable(extract_conversation)

    def test_import_top_level_includes_interceptor(self):
        """from memento import intercept works."""
        from memento import intercept, teardown_all, InterceptorHandle, GenericInterceptor
        assert callable(intercept)
        assert callable(GenericInterceptor)


class TestIntegrationOpenAI:
    """Integration test: real EtchStore + tempfile + mocked client → verify facts in DB."""

    def test_integration_facts_in_db(self, tmp_path):
        """Facts are stored in the SQLite DB and queryable."""
        import sqlite3

        db_path = tmp_path / "test_integration.db"
        from memento import EtchStore

        store = EtchStore(str(db_path), auto_migrate=True)

        completion = _make_mock_chat_completion("Hello from GPT")

        fake_module = _make_fake_openai_module()[0]
        mock_create = fake_module.OpenAI.chat.completions.create
        mock_create.return_value = completion

        with patch.dict("sys.modules", {"openai": fake_module}):
            from memento.interceptor import intercept, teardown_all

            handles = intercept(store, targets=["openai"])
            try:
                import openai
                openai.OpenAI.chat.completions.create(
                    model="gpt-4",
                    messages=[{"role": "user", "content": "Integration test"}],
                )
            finally:
                teardown_all(handles)

        store.close()

        # Read back from SQLite directly
        conn = sqlite3.connect(str(db_path))
        rows = conn.execute(
            "SELECT content, category, tags, topic_key FROM facts ORDER BY fact_id"
        ).fetchall()
        conn.close()

        contents = {r[0] for r in rows}
        assert "user: Integration test" in contents
        assert "assistant: Hello from GPT" in contents

        # Verify all facts have category=conversation
        for row in rows:
            assert row[1] == "conversation"


class TestTeardownWithRealHandles:
    """teardown_all with real interceptor handles restores correctly."""

    def test_teardown_all_restores_all(self):
        """Multiple interceptors can all be torn down."""
        fake_openai, openai_create = _make_fake_openai_module()
        fake_anthropic, anth_create = _make_fake_anthropic_module()

        with patch.dict("sys.modules", {"openai": fake_openai, "anthropic": fake_anthropic}):
            from memento.interceptor import intercept, teardown_all

            store = MagicMock(spec=EtchStore)
            store.add_fact.return_value = 1

            handles = intercept(store, targets=["openai", "anthropic"])
            assert len(handles) == 2

            # Both are patched
            assert id(fake_openai.OpenAI.chat.completions.create) != id(openai_create)
            assert id(fake_anthropic.Anthropic.messages.create) != id(anth_create)

            teardown_all(handles)

            # Both are restored
            assert id(fake_openai.OpenAI.chat.completions.create) == id(openai_create)
            assert id(fake_anthropic.Anthropic.messages.create) == id(anth_create)
