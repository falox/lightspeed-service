"""Unit tests for OLS-3460: Streaming pipeline reasoning accumulation.

Tests verify that REASONING chunks are properly accumulated into the response
variable for storage in conversation history, including proper formatting with
separators between reasoning and text content.
"""

import json
from unittest.mock import patch

import pytest

from ols import config, constants

# needs to be setup there before is_user_authorized is imported
config.ols_config.authentication_config.module = "k8s"

from ols.app.endpoints.streaming_ols import (  # noqa:E402
    LLM_REASONING_EVENT,
    response_processing_wrapper,
)
from ols.app.models.models import (  # noqa:E402
    LLMRequest,
    StreamChunkType,
    StreamedChunk,
)
from ols.utils import suid  # noqa:E402

conversation_id = suid.get_suid()


async def drain_generator(generator) -> list[str]:
    """Drain the async generator and return all yielded items."""
    return [item async for item in generator]


@pytest.fixture(scope="function")
def _load_config():
    """Load config before unit tests."""
    config.reload_from_yaml_file("tests/config/test_app_endpoints.yaml")


class TestReasoningAccumulation:
    """Test that REASONING chunks are accumulated into response variable."""

    @pytest.mark.asyncio
    async def test_reasoning_chunks_accumulated_into_response(
        self, _load_config
    ) -> None:
        """Verify REASONING chunks are accumulated into response for storage.

        OLS-3460: Reasoning tokens should be preserved in stored conversation
        response so they're available to the model on follow-up turns.
        """

        async def _fake_generator():
            # Simulate reasoning chunks from vLLM provider
            yield StreamedChunk(type=StreamChunkType.REASONING, text="Let me ")
            yield StreamedChunk(type=StreamChunkType.REASONING, text="think ")
            yield StreamedChunk(type=StreamChunkType.REASONING, text="about this.")
            # Then text chunks
            yield StreamedChunk(type=StreamChunkType.TEXT, text="The answer ")
            yield StreamedChunk(type=StreamChunkType.TEXT, text="is yes.")
            yield StreamedChunk(
                type=StreamChunkType.END,
                data={"rag_chunks": [], "truncated": False, "token_counter": None},
            )

        llm_request = LLMRequest(query="test")

        # Capture the response passed to store_data
        captured_response = {}

        def _capture_store_data(
            user_id,
            conversation_id,
            llm_request,
            response,
            tool_calls,
            tool_results,
            attachments,
            query_without_attachments,
            rag_chunks,
            history_truncated,
            timestamps,
            skip_user_id_check,
        ):
            captured_response["response"] = response

        with patch(
            "ols.app.endpoints.streaming_ols.store_data",
            side_effect=_capture_store_data,
        ):
            await drain_generator(
                response_processing_wrapper(
                    _fake_generator(),
                    user_id="test-user",
                    conversation_id=conversation_id,
                    llm_request=llm_request,
                    attachments=[],
                    query_without_attachments="test",
                    media_type=constants.MEDIA_TYPE_TEXT,
                    timestamps={},
                    skip_user_id_check=True,
                )
            )

        # Verify response contains both reasoning and text
        response = captured_response.get("response", "")
        assert "Let me think about this." in response
        assert "The answer is yes." in response

    @pytest.mark.asyncio
    async def test_reasoning_and_text_have_proper_separators(
        self, _load_config
    ) -> None:
        """Verify reasoning and text are separated properly in accumulated response.

        This test specifically checks for separators to prevent merged words.
        """

        async def _fake_generator():
            # Chunks without spaces to test separator logic
            yield StreamedChunk(type=StreamChunkType.REASONING, text="thought1")
            yield StreamedChunk(type=StreamChunkType.TEXT, text="answer1")
            yield StreamedChunk(type=StreamChunkType.REASONING, text="thought2")
            yield StreamedChunk(type=StreamChunkType.TEXT, text="answer2")
            yield StreamedChunk(
                type=StreamChunkType.END,
                data={"rag_chunks": [], "truncated": False, "token_counter": None},
            )

        llm_request = LLMRequest(query="test")
        captured_response = {}

        def _capture_store_data(
            user_id,
            conversation_id,
            llm_request,
            response,
            tool_calls,
            tool_results,
            attachments,
            query_without_attachments,
            rag_chunks,
            history_truncated,
            timestamps,
            skip_user_id_check,
        ):
            captured_response["response"] = response

        with patch(
            "ols.app.endpoints.streaming_ols.store_data",
            side_effect=_capture_store_data,
        ):
            await drain_generator(
                response_processing_wrapper(
                    _fake_generator(),
                    user_id="test-user",
                    conversation_id=conversation_id,
                    llm_request=llm_request,
                    attachments=[],
                    query_without_attachments="test",
                    media_type=constants.MEDIA_TYPE_TEXT,
                    timestamps={},
                    skip_user_id_check=True,
                )
            )

        response = captured_response.get("response", "")

        # Critical: Verify separators exist to prevent word merging
        # e.g., "thought1\n\nanswer1\n\nthought2\n\nanswer2"
        assert "thought1" in response
        assert "answer1" in response
        assert "thought2" in response
        assert "answer2" in response

        # Verify NO merged words (without separator)
        assert "thought1answer1" not in response
        assert "answer1thought2" not in response
        assert "thought2answer2" not in response

        # Verify separators are present
        assert "\n\n" in response

    @pytest.mark.asyncio
    async def test_reasoning_streamed_to_client_and_accumulated(
        self, _load_config
    ) -> None:
        """Verify reasoning is both streamed to client AND accumulated for storage.

        OLS-3460 should NOT break client-side streaming while adding storage.
        """

        async def _fake_generator():
            yield StreamedChunk(type=StreamChunkType.REASONING, text="thinking...")
            yield StreamedChunk(type=StreamChunkType.TEXT, text="answer")
            yield StreamedChunk(
                type=StreamChunkType.END,
                data={"rag_chunks": [], "truncated": False, "token_counter": None},
            )

        llm_request = LLMRequest(query="test")
        captured_response = {}
        streamed_events = []

        def _capture_store_data(
            user_id,
            conversation_id,
            llm_request,
            response,
            tool_calls,
            tool_results,
            attachments,
            query_without_attachments,
            rag_chunks,
            history_truncated,
            timestamps,
            skip_user_id_check,
        ):
            captured_response["response"] = response

        with patch(
            "ols.app.endpoints.streaming_ols.store_data",
            side_effect=_capture_store_data,
        ):
            events = await drain_generator(
                response_processing_wrapper(
                    _fake_generator(),
                    user_id="test-user",
                    conversation_id=conversation_id,
                    llm_request=llm_request,
                    attachments=[],
                    query_without_attachments="test",
                    media_type=constants.MEDIA_TYPE_JSON,
                    timestamps={},
                    skip_user_id_check=True,
                )
            )

            for item in events:
                if item.startswith("data: "):
                    parsed = json.loads(item[6:].strip())
                    streamed_events.append(parsed)

        # Verify reasoning was streamed to client
        reasoning_events = [
            e for e in streamed_events if e.get("event") == LLM_REASONING_EVENT
        ]
        assert len(reasoning_events) > 0
        assert reasoning_events[0]["data"]["reasoning"] == "thinking..."

        # Verify reasoning was accumulated for storage
        response = captured_response.get("response", "")
        assert "thinking..." in response
        assert "answer" in response

    @pytest.mark.asyncio
    async def test_empty_reasoning_chunks_handled(self, _load_config) -> None:
        """Verify empty reasoning chunks don't cause issues."""

        async def _fake_generator():
            yield StreamedChunk(type=StreamChunkType.REASONING, text="")
            yield StreamedChunk(type=StreamChunkType.TEXT, text="answer")
            yield StreamedChunk(
                type=StreamChunkType.END,
                data={"rag_chunks": [], "truncated": False, "token_counter": None},
            )

        llm_request = LLMRequest(query="test")
        captured_response = {}

        def _capture_store_data(
            user_id,
            conversation_id,
            llm_request,
            response,
            tool_calls,
            tool_results,
            attachments,
            query_without_attachments,
            rag_chunks,
            history_truncated,
            timestamps,
            skip_user_id_check,
        ):
            captured_response["response"] = response

        with patch(
            "ols.app.endpoints.streaming_ols.store_data",
            side_effect=_capture_store_data,
        ):
            await drain_generator(
                response_processing_wrapper(
                    _fake_generator(),
                    user_id="test-user",
                    conversation_id=conversation_id,
                    llm_request=llm_request,
                    attachments=[],
                    query_without_attachments="test",
                    media_type=constants.MEDIA_TYPE_TEXT,
                    timestamps={},
                    skip_user_id_check=True,
                )
            )

        response = captured_response.get("response", "")
        assert "answer" in response

    @pytest.mark.asyncio
    async def test_only_reasoning_chunks_accumulated(self, _load_config) -> None:
        """Verify response with only reasoning chunks (no text)."""

        async def _fake_generator():
            yield StreamedChunk(type=StreamChunkType.REASONING, text="deep ")
            yield StreamedChunk(type=StreamChunkType.REASONING, text="thought")
            yield StreamedChunk(
                type=StreamChunkType.END,
                data={"rag_chunks": [], "truncated": False, "token_counter": None},
            )

        llm_request = LLMRequest(query="test")
        captured_response = {}

        def _capture_store_data(
            user_id,
            conversation_id,
            llm_request,
            response,
            tool_calls,
            tool_results,
            attachments,
            query_without_attachments,
            rag_chunks,
            history_truncated,
            timestamps,
            skip_user_id_check,
        ):
            captured_response["response"] = response

        with patch(
            "ols.app.endpoints.streaming_ols.store_data",
            side_effect=_capture_store_data,
        ):
            await drain_generator(
                response_processing_wrapper(
                    _fake_generator(),
                    user_id="test-user",
                    conversation_id=conversation_id,
                    llm_request=llm_request,
                    attachments=[],
                    query_without_attachments="test",
                    media_type=constants.MEDIA_TYPE_TEXT,
                    timestamps={},
                    skip_user_id_check=True,
                )
            )

        response = captured_response.get("response", "")
        assert "deep" in response
        assert "thought" in response

    @pytest.mark.asyncio
    async def test_reasoning_with_text_media_type(self, _load_config) -> None:
        """Verify reasoning accumulation works with text media type."""

        async def _fake_generator():
            yield StreamedChunk(type=StreamChunkType.REASONING, text="thinking")
            yield StreamedChunk(type=StreamChunkType.TEXT, text="result")
            yield StreamedChunk(
                type=StreamChunkType.END,
                data={"rag_chunks": [], "truncated": False, "token_counter": None},
            )

        llm_request = LLMRequest(query="test")
        captured_response = {}

        def _capture_store_data(
            user_id,
            conversation_id,
            llm_request,
            response,
            tool_calls,
            tool_results,
            attachments,
            query_without_attachments,
            rag_chunks,
            history_truncated,
            timestamps,
            skip_user_id_check,
        ):
            captured_response["response"] = response

        with patch(
            "ols.app.endpoints.streaming_ols.store_data",
            side_effect=_capture_store_data,
        ):
            await drain_generator(
                response_processing_wrapper(
                    _fake_generator(),
                    user_id="test-user",
                    conversation_id=conversation_id,
                    llm_request=llm_request,
                    attachments=[],
                    query_without_attachments="test",
                    media_type=constants.MEDIA_TYPE_TEXT,
                    timestamps={},
                    skip_user_id_check=True,
                )
            )

        response = captured_response.get("response", "")
        assert "thinking" in response
        assert "result" in response
        assert "\n\n" in response  # Verify separator

    @pytest.mark.asyncio
    async def test_reasoning_with_json_media_type(self, _load_config) -> None:
        """Verify reasoning accumulation works with JSON media type."""

        async def _fake_generator():
            yield StreamedChunk(type=StreamChunkType.REASONING, text="analyze")
            yield StreamedChunk(type=StreamChunkType.TEXT, text="conclusion")
            yield StreamedChunk(
                type=StreamChunkType.END,
                data={"rag_chunks": [], "truncated": False, "token_counter": None},
            )

        llm_request = LLMRequest(query="test")
        captured_response = {}

        def _capture_store_data(
            user_id,
            conversation_id,
            llm_request,
            response,
            tool_calls,
            tool_results,
            attachments,
            query_without_attachments,
            rag_chunks,
            history_truncated,
            timestamps,
            skip_user_id_check,
        ):
            captured_response["response"] = response

        with patch(
            "ols.app.endpoints.streaming_ols.store_data",
            side_effect=_capture_store_data,
        ):
            await drain_generator(
                response_processing_wrapper(
                    _fake_generator(),
                    user_id="test-user",
                    conversation_id=conversation_id,
                    llm_request=llm_request,
                    attachments=[],
                    query_without_attachments="test",
                    media_type=constants.MEDIA_TYPE_JSON,
                    timestamps={},
                    skip_user_id_check=True,
                )
            )

        response = captured_response.get("response", "")
        assert "analyze" in response
        assert "conclusion" in response
        assert "\n\n" in response  # Verify separator

    @pytest.mark.asyncio
    async def test_empty_chunks_dont_create_extra_separators(
        self, _load_config
    ) -> None:
        """Verify empty chunks don't create triple newlines or other artifacts."""

        async def _fake_generator():
            # Test empty reasoning chunk
            yield StreamedChunk(type=StreamChunkType.REASONING, text="")
            yield StreamedChunk(type=StreamChunkType.TEXT, text="answer")
            # Test reasoning with only newline
            yield StreamedChunk(type=StreamChunkType.REASONING, text="\n")
            yield StreamedChunk(type=StreamChunkType.TEXT, text="more")
            yield StreamedChunk(
                type=StreamChunkType.END,
                data={"rag_chunks": [], "truncated": False, "token_counter": None},
            )

        llm_request = LLMRequest(query="test")
        captured_response = {}

        def _capture_store_data(
            user_id,
            conversation_id,
            llm_request,
            response,
            tool_calls,
            tool_results,
            attachments,
            query_without_attachments,
            rag_chunks,
            history_truncated,
            timestamps,
            skip_user_id_check,
        ):
            captured_response["response"] = response

        with patch(
            "ols.app.endpoints.streaming_ols.store_data",
            side_effect=_capture_store_data,
        ):
            await drain_generator(
                response_processing_wrapper(
                    _fake_generator(),
                    user_id="test-user",
                    conversation_id=conversation_id,
                    llm_request=llm_request,
                    attachments=[],
                    query_without_attachments="test",
                    media_type=constants.MEDIA_TYPE_TEXT,
                    timestamps={},
                    skip_user_id_check=True,
                )
            )

        response = captured_response.get("response", "")
        # Should not have triple newlines
        assert "\n\n\n" not in response
        # Should have content
        assert "answer" in response
        assert "more" in response

    @pytest.mark.asyncio
    async def test_special_characters_and_unicode(self, _load_config) -> None:
        """Verify special characters and unicode are preserved correctly."""

        async def _fake_generator():
            # Unicode reasoning
            yield StreamedChunk(
                type=StreamChunkType.REASONING,
                text="考えている",  # Japanese: "thinking"
            )
            # Text with special chars
            yield StreamedChunk(type=StreamChunkType.TEXT, text="Answer: 42 (✓) [OK]")
            yield StreamedChunk(
                type=StreamChunkType.END,
                data={"rag_chunks": [], "truncated": False, "token_counter": None},
            )

        llm_request = LLMRequest(query="test")
        captured_response = {}

        def _capture_store_data(
            user_id,
            conversation_id,
            llm_request,
            response,
            tool_calls,
            tool_results,
            attachments,
            query_without_attachments,
            rag_chunks,
            history_truncated,
            timestamps,
            skip_user_id_check,
        ):
            captured_response["response"] = response

        with patch(
            "ols.app.endpoints.streaming_ols.store_data",
            side_effect=_capture_store_data,
        ):
            await drain_generator(
                response_processing_wrapper(
                    _fake_generator(),
                    user_id="test-user",
                    conversation_id=conversation_id,
                    llm_request=llm_request,
                    attachments=[],
                    query_without_attachments="test",
                    media_type=constants.MEDIA_TYPE_TEXT,
                    timestamps={},
                    skip_user_id_check=True,
                )
            )

        response = captured_response.get("response", "")
        # Verify unicode preserved
        assert "考えている" in response
        # Verify special chars preserved
        assert "✓" in response
        assert "[OK]" in response
        # Verify separator exists
        assert "\n\n" in response


class TestErrorHandling:
    """Test error handling paths in reasoning accumulation."""

    @pytest.mark.asyncio
    async def test_generator_fails_mid_stream_reasoning(self, _load_config) -> None:
        """Verify response accumulation handles generator failure mid-stream."""

        async def _failing_generator():
            yield StreamedChunk(type=StreamChunkType.REASONING, text="thinking")
            raise RuntimeError("Simulated generator failure")

        llm_request = LLMRequest(query="test")

        def _capture_store_data(*args, **kwargs):
            # store_data not called on error
            pass

        with patch(
            "ols.app.endpoints.streaming_ols.store_data",
            side_effect=_capture_store_data,
        ):
            events = await drain_generator(
                response_processing_wrapper(
                    _failing_generator(),
                    user_id="test-user",
                    conversation_id=conversation_id,
                    llm_request=llm_request,
                    attachments=[],
                    query_without_attachments="test",
                    media_type=constants.MEDIA_TYPE_TEXT,
                    timestamps={},
                    skip_user_id_check=True,
                )
            )

            # Should get error event
            error_found = False
            for event in events:
                if "error" in event.lower():
                    error_found = True
                    break

            assert error_found, "Expected error event in stream"

    @pytest.mark.asyncio
    async def test_store_data_failure_doesnt_lose_response(self, _load_config) -> None:
        """Verify that store_data failure produces error response."""

        async def _fake_generator():
            yield StreamedChunk(type=StreamChunkType.REASONING, text="reasoning")
            yield StreamedChunk(type=StreamChunkType.TEXT, text="text")
            yield StreamedChunk(
                type=StreamChunkType.END,
                data={"rag_chunks": [], "truncated": False, "token_counter": None},
            )

        llm_request = LLMRequest(query="test")

        def _failing_store_data(*args, **kwargs):
            raise RuntimeError("Database connection lost")

        with patch(
            "ols.app.endpoints.streaming_ols.store_data",
            side_effect=_failing_store_data,
        ):
            events = await drain_generator(
                response_processing_wrapper(
                    _fake_generator(),
                    user_id="test-user",
                    conversation_id=conversation_id,
                    llm_request=llm_request,
                    attachments=[],
                    query_without_attachments="test",
                    media_type=constants.MEDIA_TYPE_JSON,
                    timestamps={},
                    skip_user_id_check=True,
                )
            )

            # Should include reasoning in streamed events before error
            has_reasoning = False
            has_text = False
            has_error = False

            for event in events:
                if event.startswith("data: "):
                    try:
                        parsed = json.loads(event[6:].strip())
                        if parsed.get("event") == "reasoning":
                            has_reasoning = True
                        if parsed.get("event") == "token":
                            has_text = True
                        if parsed.get("event") == "error":
                            has_error = True
                    except json.JSONDecodeError:
                        pass

            assert has_reasoning, "Reasoning should be streamed before error"
            assert has_text, "Text should be streamed before error"
            assert has_error, "Error should be present"
