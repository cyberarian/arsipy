from functools import wraps
from typing import Any, Callable
import time

class CacheManager:
    def __init__(self):
        self._cache = {}
        
    def cache_query(self, ttl: int = 3600) -> Callable:
        """
        Cache decorator with time-to-live (TTL)
        
        Args:
            ttl: Time to live in seconds (default: 1 hour)
        """
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(*args, **kwargs) -> Any:
                # Create cache key from function name and arguments
                cache_key = f"{func.__name__}:{str(args)}:{str(kwargs)}"
                
                # Check if cached and not expired
                if cache_key in self._cache:
                    result, timestamp = self._cache[cache_key]
                    if time.time() - timestamp < ttl:
                        return result
                
                # Execute function and cache result
                result = func(*args, **kwargs)
                self._cache[cache_key] = (result, time.time())
                return result
                
            return wrapper
        return decorator
