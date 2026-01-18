import asyncio
import time
import logging

logger = logging.getLogger(__name__)

class AsyncTokenBucket:
    """
    An asynchronous token bucket rate limiter.
    
    Tokens are refilled at a fixed rate (tokens per second) up to a maximum capacity.
    Consumers must acquire a token before proceeding. If no tokens are available,
    they will wait until one becomes available.
    """
    
    def __init__(self, rate: float, capacity: float):
        """
        Initialize the token bucket.
        
        Args:
            rate: The rate at which tokens are added to the bucket (tokens/second).
            capacity: The maximum number of tokens the bucket can hold.
        """
        self.rate = rate
        self.capacity = capacity
        self.tokens = capacity
        self.last_refill_timestamp = time.monotonic()
        self._lock = asyncio.Lock()
        
    def _refill(self):
        """Refill tokens based on time elapsed since last refill."""
        now = time.monotonic()
        elapsed = now - self.last_refill_timestamp
        new_tokens = elapsed * self.rate
        
        if new_tokens > 0:
            self.tokens = min(self.capacity, self.tokens + new_tokens)
            self.last_refill_timestamp = now
            
    async def acquire(self):
        """
        Acquire a token from the bucket. Blocks until a token is available.
        """
        while True:
            async with self._lock:
                self._refill()
                
                if self.tokens >= 1:
                    self.tokens -= 1
                    return
                
                missing = 1.0 - self.tokens
                wait_time = missing / self.rate
            await asyncio.sleep(wait_time + 0.01)

    async def update_rate(self, new_rate: float):
        """Update the refill rate dynamically."""
        async with self._lock:
            self._refill()
            self.rate = new_rate
            logger.info(f"Rate limiter updated to {new_rate} RPS")
