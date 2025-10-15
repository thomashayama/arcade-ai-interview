"""
Tests for the OpenAICache class.
"""

import pytest
import json
import pickle
import tempfile
import shutil
from pathlib import Path
from datetime import datetime

from utils import OpenAICache


class TestOpenAICache:
    """Test suite for OpenAICache class."""

    @pytest.fixture
    def temp_cache_dir(self):
        """Create a temporary cache directory for testing."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        # Cleanup
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def cache(self, temp_cache_dir):
        """Create an OpenAICache instance with temporary directory."""
        return OpenAICache(cache_dir=temp_cache_dir)

    def test_cache_initialization(self, temp_cache_dir, cache):
        """Test that cache directories are created on initialization."""
        assert cache.cache_dir.exists()
        assert cache.text_cache_dir.exists()
        assert cache.image_cache_dir.exists()
        assert cache.cache_dir == Path(temp_cache_dir)

    def test_generate_cache_key_deterministic(self, cache):
        """Test that same params always generate same cache key."""
        params1 = {"model": "gpt-4", "temperature": 0.7, "messages": [{"role": "user", "content": "test"}]}
        params2 = {"model": "gpt-4", "temperature": 0.7, "messages": [{"role": "user", "content": "test"}]}

        key1 = cache._generate_cache_key(params1)
        key2 = cache._generate_cache_key(params2)

        assert key1 == key2
        assert len(key1) == 64  # SHA256 hex digest length

    def test_generate_cache_key_different_params(self, cache):
        """Test that different params generate different cache keys."""
        params1 = {"model": "gpt-4", "temperature": 0.7}
        params2 = {"model": "gpt-4", "temperature": 0.8}

        key1 = cache._generate_cache_key(params1)
        key2 = cache._generate_cache_key(params2)

        assert key1 != key2

    def test_generate_cache_key_order_independent(self, cache):
        """Test that parameter order doesn't affect cache key."""
        params1 = {"model": "gpt-4", "temperature": 0.7, "max_tokens": 100}
        params2 = {"max_tokens": 100, "temperature": 0.7, "model": "gpt-4"}

        key1 = cache._generate_cache_key(params1)
        key2 = cache._generate_cache_key(params2)

        assert key1 == key2

    def test_get_cache_path_text(self, cache):
        """Test cache path generation for text responses."""
        cache_key = "abc123"
        path = cache._get_cache_path(cache_key, "text")

        assert path.parent == cache.text_cache_dir
        assert path.name == f"{cache_key}.json"

    def test_get_cache_path_image(self, cache):
        """Test cache path generation for image responses."""
        cache_key = "xyz789"
        path = cache._get_cache_path(cache_key, "images")

        assert path.parent == cache.image_cache_dir
        assert path.name == f"{cache_key}.pkl"

    def test_set_and_get_text_cache(self, cache):
        """Test storing and retrieving text responses."""
        request_params = {"model": "gpt-4", "messages": [{"role": "user", "content": "Hello"}]}
        response = {"choices": [{"message": {"content": "Hi there!"}}]}

        # Set cache
        cache.set(request_params, response, cache_type="text")

        # Get cache
        cached_response = cache.get(request_params, cache_type="text")

        assert cached_response == response

    def test_set_and_get_image_cache(self, cache):
        """Test storing and retrieving image responses."""
        request_params = {"prompt": "A sunset", "size": "1024x1024"}
        response = {"data": [{"url": "https://example.com/image.png"}]}

        # Set cache
        cache.set(request_params, response, cache_type="images")

        # Get cache
        cached_response = cache.get(request_params, cache_type="images")

        assert cached_response == response

    def test_get_nonexistent_cache(self, cache):
        """Test that getting non-existent cache returns None."""
        request_params = {"model": "gpt-4", "messages": [{"role": "user", "content": "test"}]}

        result = cache.get(request_params, cache_type="text")

        assert result is None

    def test_cache_file_structure_text(self, cache):
        """Test that cached text files have correct structure."""
        request_params = {"model": "gpt-4", "temperature": 0.5}
        response = {"choices": [{"message": {"content": "Test response"}}]}

        cache.set(request_params, response, cache_type="text")

        # Get the cache file path
        cache_key = cache._generate_cache_key(request_params)
        cache_path = cache._get_cache_path(cache_key, "text")

        # Read and verify structure
        with open(cache_path, 'r', encoding='utf-8') as f:
            cached_data = json.load(f)

        assert 'timestamp' in cached_data
        assert 'request_params' in cached_data
        assert 'response' in cached_data
        assert cached_data['request_params'] == request_params
        assert cached_data['response'] == response

        # Verify timestamp is valid ISO format
        datetime.fromisoformat(cached_data['timestamp'])

    def test_cache_file_structure_image(self, cache):
        """Test that cached image files have correct structure."""
        request_params = {"prompt": "A cat", "size": "512x512"}
        response = {"data": [{"url": "https://example.com/cat.png"}]}

        cache.set(request_params, response, cache_type="images")

        # Get the cache file path
        cache_key = cache._generate_cache_key(request_params)
        cache_path = cache._get_cache_path(cache_key, "images")

        # Read and verify structure
        with open(cache_path, 'rb') as f:
            cached_data = pickle.load(f)

        assert 'timestamp' in cached_data
        assert 'request_params' in cached_data
        assert 'response' in cached_data
        assert cached_data['response'] == response

    def test_clear_all_cache(self, cache):
        """Test clearing all cached items."""
        # Add some cached items
        cache.set({"model": "gpt-4"}, {"result": "text1"}, cache_type="text")
        cache.set({"model": "gpt-3.5"}, {"result": "text2"}, cache_type="text")
        cache.set({"prompt": "test"}, {"result": "image1"}, cache_type="images")

        # Clear all
        deleted_count = cache.clear()

        assert deleted_count == 3
        assert len(list(cache.text_cache_dir.glob("*.json"))) == 0
        assert len(list(cache.image_cache_dir.glob("*.pkl"))) == 0

    def test_clear_text_cache_only(self, cache):
        """Test clearing only text cached items."""
        # Add some cached items
        cache.set({"model": "gpt-4"}, {"result": "text1"}, cache_type="text")
        cache.set({"prompt": "test"}, {"result": "image1"}, cache_type="images")

        # Clear text only
        deleted_count = cache.clear(cache_type="text")

        assert deleted_count == 1
        assert len(list(cache.text_cache_dir.glob("*.json"))) == 0
        assert len(list(cache.image_cache_dir.glob("*.pkl"))) == 1

    def test_clear_image_cache_only(self, cache):
        """Test clearing only image cached items."""
        # Add some cached items
        cache.set({"model": "gpt-4"}, {"result": "text1"}, cache_type="text")
        cache.set({"prompt": "test"}, {"result": "image1"}, cache_type="images")

        # Clear images only
        deleted_count = cache.clear(cache_type="images")

        assert deleted_count == 1
        assert len(list(cache.text_cache_dir.glob("*.json"))) == 1
        assert len(list(cache.image_cache_dir.glob("*.pkl"))) == 0

    def test_get_stats_empty(self, cache):
        """Test cache statistics with empty cache."""
        stats = cache.get_stats()

        assert stats['text_cache_count'] == 0
        assert stats['image_cache_count'] == 0
        assert stats['total_cached_items'] == 0
        assert stats['total_size_bytes'] == 0
        assert stats['total_size_mb'] == 0

    def test_get_stats_with_items(self, cache):
        """Test cache statistics with cached items."""
        # Add some cached items
        cache.set({"model": "gpt-4"}, {"result": "text response"}, cache_type="text")
        cache.set({"prompt": "test"}, {"data": [{"url": "example.com"}]}, cache_type="images")

        stats = cache.get_stats()

        assert stats['text_cache_count'] == 1
        assert stats['image_cache_count'] == 1
        assert stats['total_cached_items'] == 2
        assert stats['total_size_bytes'] > 0
        assert stats['total_size_mb'] >= 0  # Small files may round to 0.0 MB

    def test_cache_with_complex_nested_data(self, cache):
        """Test caching with deeply nested request parameters."""
        request_params = {
            "model": "gpt-4",
            "messages": [
                {"role": "system", "content": "You are helpful"},
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi"},
                {"role": "user", "content": "How are you?"}
            ],
            "temperature": 0.7,
            "max_tokens": 1000,
            "top_p": 0.9,
            "frequency_penalty": 0.5
        }
        response = {"choices": [{"message": {"content": "I'm doing well!"}}]}

        cache.set(request_params, response, cache_type="text")
        cached_response = cache.get(request_params, cache_type="text")

        assert cached_response == response

    def test_corrupted_cache_file_handling(self, cache):
        """Test that corrupted cache files are handled gracefully."""
        request_params = {"model": "gpt-4"}

        # Create a corrupted cache file
        cache_key = cache._generate_cache_key(request_params)
        cache_path = cache._get_cache_path(cache_key, "text")

        with open(cache_path, 'w') as f:
            f.write("corrupted json data {{{")

        # Should return None for corrupted cache
        result = cache.get(request_params, cache_type="text")

        assert result is None

    def test_multiple_requests_same_params(self, cache):
        """Test that multiple requests with same params use cache."""
        request_params = {"model": "gpt-4", "temperature": 0.7}
        response = {"choices": [{"message": {"content": "Response"}}]}

        # First request - sets cache
        cache.set(request_params, response, cache_type="text")

        # Subsequent requests - should get from cache
        for _ in range(5):
            cached = cache.get(request_params, cache_type="text")
            assert cached == response

        # Should still only have one cached file
        stats = cache.get_stats()
        assert stats['text_cache_count'] == 1

    def test_cache_key_with_special_characters(self, cache):
        """Test cache key generation with special characters in params."""
        request_params = {
            "messages": [{"role": "user", "content": "Test with 特殊字符 and émojis 🎉"}],
            "model": "gpt-4"
        }
        response = {"result": "success"}

        cache.set(request_params, response, cache_type="text")
        cached = cache.get(request_params, cache_type="text")

        assert cached == response
