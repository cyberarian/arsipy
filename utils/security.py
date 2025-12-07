import time
from typing import Dict, Optional
from datetime import datetime, timedelta
import re
import hashlib
import jwt
from functools import wraps

class SecurityManager:
    def __init__(self):
        self._rate_limits: Dict[str, list] = {}
        self._max_requests = 60  # requests per minute
        
    def rate_limiter(self, client_ip: str) -> bool:
        """
        Implement rate limiting per IP
        Returns False if rate limit exceeded
        """
        current_time = time.time()
        
        # Initialize or clean old requests
        if client_ip not in self._rate_limits:
            self._rate_limits[client_ip] = []
        
        # Remove requests older than 1 minute
        self._rate_limits[client_ip] = [
            t for t in self._rate_limits[client_ip]
            if current_time - t < 60
        ]
        
        # Check rate limit
        if len(self._rate_limits[client_ip]) >= self._max_requests:
            return False
            
        # Add new request
        self._rate_limits[client_ip].append(current_time)
        return True
        
    def sanitize_input(self, text: str) -> str:
        """Remove potentially harmful characters from input"""
        # Remove any HTML/script tags
        text = re.sub(r'<[^>]*>', '', text)
        # Remove any special characters except basic punctuation
        text = re.sub(r'[^\w\s.,!?-]', '', text)
        return text.strip()
        
    def generate_session_token(self, user_id: str) -> str:
        payload = {
            'user_id': user_id,
            'exp': datetime.utcnow() + timedelta(days=1)
        }
        return jwt.encode(payload, self.secret_key, algorithm='HS256')
