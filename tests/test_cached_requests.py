"""
Tests for the cached_openai_request function.
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
import tempfile
import shutil

from utils import OpenAICache, cached_openai_request


class TestCachedOpenAIRequest:
    """Test suite for cached_openai_request function."""

    @pytest.fixture
    def temp_cache_dir(self):
        """Create a temporary cache directory for testing."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def cache(self, temp_cache_dir):
        """Create an OpenAICache instance with temporary directory."""
        return OpenAICache(cache_dir=temp_cache_dir)

    @pytest.fixture
    def mock_openai_client(self):
        """Create a mock OpenAI client."""
        client = Mock()
        return client

    def test_chat_request_cache_miss(self, mock_openai_client, cache):
        """Test chat request when cache is empty (cache miss)."""
        # Setup mock response
        mock_response = Mock()
        mock_response.model_dump.return_value = {
            "id": "chatcmpl-123",
            "choices": [{"message": {"content": "Hello, world!"}}],
            "model": "gpt-4",
            "usage": {"total_tokens": 10}
        }
        mock_openai_client.chat.completions.create.return_value = mock_response

        # Make request
        result = cached_openai_request(
            client=mock_openai_client,
            cache=cache,
            request_type="chat",
            model="gpt-4",
            messages=[{"role": "user", "content": "Hi"}]
        )

        # Verify API was called
        mock_openai_client.chat.completions.create.assert_called_once_with(
            model="gpt-4",
            messages=[{"role": "user", "content": "Hi"}]
        )

        # Verify result
        assert result == mock_response.model_dump.return_value
        assert result["choices"][0]["message"]["content"] == "Hello, world!"

    def test_chat_request_cache_hit(self, mock_openai_client, cache):
        """Test chat request when response is cached (cache hit)."""
        # Pre-populate cache
        request_params = {
            "request_type": "chat",
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "Hi"}]
        }
        cached_response = {
            "id": "chatcmpl-123",
            "choices": [{"message": {"content": "Cached response"}}]
        }
        cache.set(request_params, cached_response, cache_type="text")

        # Make request
        result = cached_openai_request(
            client=mock_openai_client,
            cache=cache,
            request_type="chat",
            model="gpt-4",
            messages=[{"role": "user", "content": "Hi"}]
        )

        # Verify API was NOT called
        mock_openai_client.chat.completions.create.assert_not_called()

        # Verify result comes from cache
        assert result == cached_response
        assert result["choices"][0]["message"]["content"] == "Cached response"

    def test_image_request_cache_miss(self, mock_openai_client, cache):
        """Test image request when cache is empty (cache miss)."""
        # Setup mock response
        mock_response = Mock()
        mock_response.model_dump.return_value = {
            "data": [{"url": "https://example.com/image.png"}],
            "created": 1234567890
        }
        mock_openai_client.images.generate.return_value = mock_response

        # Make request
        result = cached_openai_request(
            client=mock_openai_client,
            cache=cache,
            request_type="image",
            prompt="A beautiful sunset",
            size="1024x1024",
            model="dall-e-3"
        )

        # Verify API was called
        mock_openai_client.images.generate.assert_called_once_with(
            prompt="A beautiful sunset",
            size="1024x1024",
            model="dall-e-3"
        )

        # Verify result
        assert result == mock_response.model_dump.return_value
        assert result["data"][0]["url"] == "https://example.com/image.png"

    def test_image_request_cache_hit(self, mock_openai_client, cache):
        """Test image request when response is cached (cache hit)."""
        # Pre-populate cache
        request_params = {
            "request_type": "image",
            "prompt": "A cat",
            "size": "512x512"
        }
        cached_response = {
            "data": [{"url": "https://example.com/cached-cat.png"}]
        }
        cache.set(request_params, cached_response, cache_type="images")

        # Make request
        result = cached_openai_request(
            client=mock_openai_client,
            cache=cache,
            request_type="image",
            prompt="A cat",
            size="512x512"
        )

        # Verify API was NOT called
        mock_openai_client.images.generate.assert_not_called()

        # Verify result comes from cache
        assert result == cached_response
        assert result["data"][0]["url"] == "https://example.com/cached-cat.png"

    def test_multiple_identical_requests_only_one_api_call(self, mock_openai_client, cache):
        """Test that multiple identical requests only make one API call."""
        # Setup mock response
        mock_response = Mock()
        mock_response.model_dump.return_value = {
            "choices": [{"message": {"content": "Response"}}]
        }
        mock_openai_client.chat.completions.create.return_value = mock_response

        # Make the same request multiple times
        for _ in range(5):
            result = cached_openai_request(
                client=mock_openai_client,
                cache=cache,
                request_type="chat",
                model="gpt-4",
                messages=[{"role": "user", "content": "Same question"}]
            )
            assert result["choices"][0]["message"]["content"] == "Response"

        # Verify API was called only once
        assert mock_openai_client.chat.completions.create.call_count == 1

    def test_different_parameters_different_cache(self, mock_openai_client, cache):
        """Test that different parameters result in different cache entries."""
        # Setup mock response
        mock_response1 = Mock()
        mock_response1.model_dump.return_value = {
            "choices": [{"message": {"content": "Response 1"}}]
        }
        mock_response2 = Mock()
        mock_response2.model_dump.return_value = {
            "choices": [{"message": {"content": "Response 2"}}]
        }

        mock_openai_client.chat.completions.create.side_effect = [
            mock_response1,
            mock_response2
        ]

        # Make two requests with different temperatures
        result1 = cached_openai_request(
            client=mock_openai_client,
            cache=cache,
            request_type="chat",
            model="gpt-4",
            messages=[{"role": "user", "content": "Hi"}],
            temperature=0.5
        )

        result2 = cached_openai_request(
            client=mock_openai_client,
            cache=cache,
            request_type="chat",
            model="gpt-4",
            messages=[{"role": "user", "content": "Hi"}],
            temperature=0.9
        )

        # Verify both API calls were made (different cache keys)
        assert mock_openai_client.chat.completions.create.call_count == 2

        # Verify different responses
        assert result1["choices"][0]["message"]["content"] == "Response 1"
        assert result2["choices"][0]["message"]["content"] == "Response 2"

        # Verify cache has two entries
        stats = cache.get_stats()
        assert stats['text_cache_count'] == 2

    def test_unsupported_request_type_raises_error(self, mock_openai_client, cache):
        """Test that unsupported request type raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported request type"):
            cached_openai_request(
                client=mock_openai_client,
                cache=cache,
                request_type="unsupported",
                some_param="value"
            )

    def test_chat_request_with_all_parameters(self, mock_openai_client, cache):
        """Test chat request with comprehensive parameters."""
        # Setup mock response
        mock_response = Mock()
        mock_response.model_dump.return_value = {
            "choices": [{"message": {"content": "Detailed response"}}]
        }
        mock_openai_client.chat.completions.create.return_value = mock_response

        # Make request with many parameters
        result = cached_openai_request(
            client=mock_openai_client,
            cache=cache,
            request_type="chat",
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are helpful"},
                {"role": "user", "content": "Hello"}
            ],
            temperature=0.7,
            max_tokens=1000,
            top_p=0.9,
            frequency_penalty=0.5,
            presence_penalty=0.5,
            n=1
        )

        # Verify all parameters were passed to API
        call_kwargs = mock_openai_client.chat.completions.create.call_args.kwargs
        assert call_kwargs["model"] == "gpt-4"
        assert call_kwargs["temperature"] == 0.7
        assert call_kwargs["max_tokens"] == 1000
        assert call_kwargs["top_p"] == 0.9
        assert call_kwargs["frequency_penalty"] == 0.5
        assert call_kwargs["presence_penalty"] == 0.5
        assert call_kwargs["n"] == 1

        # Verify result
        assert result["choices"][0]["message"]["content"] == "Detailed response"

    def test_image_request_with_all_parameters(self, mock_openai_client, cache):
        """Test image request with comprehensive parameters."""
        # Setup mock response
        mock_response = Mock()
        mock_response.model_dump.return_value = {
            "data": [{"url": "https://example.com/detailed-image.png"}]
        }
        mock_openai_client.images.generate.return_value = mock_response

        # Make request with many parameters
        result = cached_openai_request(
            client=mock_openai_client,
            cache=cache,
            request_type="image",
            model="dall-e-3",
            prompt="A detailed landscape",
            size="1024x1024",
            quality="hd",
            n=1
        )

        # Verify all parameters were passed to API
        call_kwargs = mock_openai_client.images.generate.call_args.kwargs
        assert call_kwargs["model"] == "dall-e-3"
        assert call_kwargs["prompt"] == "A detailed landscape"
        assert call_kwargs["size"] == "1024x1024"
        assert call_kwargs["quality"] == "hd"
        assert call_kwargs["n"] == 1

        # Verify result
        assert result["data"][0]["url"] == "https://example.com/detailed-image.png"

    def test_cache_persists_between_function_calls(self, mock_openai_client, temp_cache_dir):
        """Test that cache persists between different cache instances."""
        # Setup mock response
        mock_response = Mock()
        mock_response.model_dump.return_value = {
            "choices": [{"message": {"content": "Persistent response"}}]
        }
        mock_openai_client.chat.completions.create.return_value = mock_response

        # First cache instance - make request
        cache1 = OpenAICache(cache_dir=temp_cache_dir)
        result1 = cached_openai_request(
            client=mock_openai_client,
            cache=cache1,
            request_type="chat",
            model="gpt-4",
            messages=[{"role": "user", "content": "Test"}]
        )

        # Second cache instance - should read from persisted cache
        cache2 = OpenAICache(cache_dir=temp_cache_dir)
        result2 = cached_openai_request(
            client=mock_openai_client,
            cache=cache2,
            request_type="chat",
            model="gpt-4",
            messages=[{"role": "user", "content": "Test"}]
        )

        # Verify API was called only once
        assert mock_openai_client.chat.completions.create.call_count == 1

        # Verify both results are identical
        assert result1 == result2
        assert result2["choices"][0]["message"]["content"] == "Persistent response"
