"""
Utility functions for caching OpenAI API responses to manage costs and rate limits.
"""

import json
import hashlib
from pathlib import Path
from typing import Any, Optional, Dict
from datetime import datetime
import pickle


class OpenAICache:
    """
    Cache manager for OpenAI API responses.

    Stores responses based on a hash of the request parameters to avoid
    redundant API calls during development and testing.
    """

    def __init__(self, cache_dir: str = ".cache"):
        """
        Initialize the cache manager.

        Args:
            cache_dir: Directory to store cached responses (default: .cache)
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)

        # Create subdirectories for different cache types
        self.text_cache_dir = self.cache_dir / "text"
        self.image_cache_dir = self.cache_dir / "images"

        self.text_cache_dir.mkdir(exist_ok=True)
        self.image_cache_dir.mkdir(exist_ok=True)

    def _generate_cache_key(self, request_params: Dict[str, Any]) -> str:
        """
        Generate a unique cache key based on request parameters.

        Args:
            request_params: Dictionary of API request parameters

        Returns:
            SHA256 hash of the serialized parameters
        """
        # Convert params to a canonical JSON string
        canonical_json = json.dumps(request_params, sort_keys=True)

        # Generate hash
        hash_obj = hashlib.sha256(canonical_json.encode('utf-8'))
        return hash_obj.hexdigest()

    def _get_cache_path(self, cache_key: str, cache_type: str = "text") -> Path:
        """
        Get the file path for a cached item.

        Args:
            cache_key: The cache key (hash)
            cache_type: Type of cache ("text" or "images")

        Returns:
            Path to the cache file
        """
        if cache_type == "text":
            return self.text_cache_dir / f"{cache_key}.json"
        else:
            return self.image_cache_dir / f"{cache_key}.pkl"

    def get(self, request_params: Dict[str, Any], cache_type: str = "text") -> Optional[Any]:
        """
        Retrieve a cached response if it exists.

        Args:
            request_params: Dictionary of API request parameters
            cache_type: Type of cache ("text" or "images")

        Returns:
            Cached response if found, None otherwise
        """
        cache_key = self._generate_cache_key(request_params)
        cache_path = self._get_cache_path(cache_key, cache_type)

        if cache_path.exists():
            try:
                if cache_type == "text":
                    with open(cache_path, 'r', encoding='utf-8') as f:
                        cached_data = json.load(f)
                else:
                    with open(cache_path, 'rb') as f:
                        cached_data = pickle.load(f)

                print(f"Cache hit for {cache_type} request (key: {cache_key[:8]}...)")
                return cached_data['response']
            except (json.JSONDecodeError, pickle.PickleError, KeyError) as e:
                print(f"Cache file corrupted, will regenerate: {e}")
                return None

        print(f"Cache miss for {cache_type} request (key: {cache_key[:8]}...)")
        return None

    def set(self, request_params: Dict[str, Any], response: Any, cache_type: str = "text") -> None:
        """
        Store a response in the cache.

        Args:
            request_params: Dictionary of API request parameters
            response: The API response to cache
            cache_type: Type of cache ("text" or "images")
        """
        cache_key = self._generate_cache_key(request_params)
        cache_path = self._get_cache_path(cache_key, cache_type)

        cached_data = {
            'timestamp': datetime.now().isoformat(),
            'request_params': request_params,
            'response': response
        }

        try:
            if cache_type == "text":
                with open(cache_path, 'w', encoding='utf-8') as f:
                    json.dump(cached_data, f, indent=2)
            else:
                with open(cache_path, 'wb') as f:
                    pickle.dump(cached_data, f)

            print(f"Cached {cache_type} response (key: {cache_key[:8]}...)")
        except Exception as e:
            print(f"Failed to cache response: {e}")

    def clear(self, cache_type: Optional[str] = None) -> int:
        """
        Clear cached responses.

        Args:
            cache_type: Type to clear ("text", "images", or None for all)

        Returns:
            Number of files deleted
        """
        deleted_count = 0

        if cache_type in (None, "text"):
            for cache_file in self.text_cache_dir.glob("*.json"):
                cache_file.unlink()
                deleted_count += 1

        if cache_type in (None, "images"):
            for cache_file in self.image_cache_dir.glob("*.pkl"):
                cache_file.unlink()
                deleted_count += 1

        print(f"Cleared {deleted_count} cached responses")
        return deleted_count

    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dictionary with cache statistics
        """
        text_files = list(self.text_cache_dir.glob("*.json"))
        image_files = list(self.image_cache_dir.glob("*.pkl"))

        total_size = sum(f.stat().st_size for f in text_files + image_files)

        return {
            'text_cache_count': len(text_files),
            'image_cache_count': len(image_files),
            'total_cached_items': len(text_files) + len(image_files),
            'total_size_bytes': total_size,
            'total_size_mb': round(total_size / (1024 * 1024), 2)
        }


def cached_openai_request(
    client: Any,
    cache: OpenAICache,
    request_type: str,
    **request_params
) -> Any:
    """
    Make an OpenAI API request with caching.

    Args:
        client: OpenAI client instance
        cache: OpenAICache instance
        request_type: Type of request ("chat", "image", etc.)
        **request_params: Parameters to pass to the API

    Returns:
        API response (from cache or fresh)
    """
    # Determine cache type based on request
    cache_type = "images" if request_type == "image" else "text"

    # Add request type to params for unique caching
    cache_params = {
        'request_type': request_type,
        **request_params
    }

    # Try to get from cache
    cached_response = cache.get(cache_params, cache_type=cache_type)
    if cached_response is not None:
        return cached_response

    # Make the actual API request
    print(f" Making fresh {request_type} API request...")

    if request_type == "chat":
        response = client.chat.completions.create(**request_params)
        # Convert to dict for caching
        response_dict = response.model_dump()
    elif request_type == "image":
        response = client.images.generate(**request_params)
        response_dict = response.model_dump()
    else:
        raise ValueError(f"Unsupported request type: {request_type}")

    # Cache the response
    cache.set(cache_params, response_dict, cache_type=cache_type)

    return response_dict


# Example usage:
if __name__ == "__main__":
    # Initialize cache
    cache = OpenAICache()

    # Show cache stats
    stats = cache.get_stats()
    print(f"\nCache Statistics:")
    print(f"  Text responses: {stats['text_cache_count']}")
    print(f"  Image responses: {stats['image_cache_count']}")
    print(f"  Total size: {stats['total_size_mb']} MB")

    # Example of clearing cache (uncomment to use)
    # cache.clear()  # Clear all
    # cache.clear("text")  # Clear only text responses
    # cache.clear("images")  # Clear only image responses
