import time
import logging
from functools import wraps
from typing import Any, Callable

logger = logging.getLogger(__name__)

class SystemMonitor:
    def __init__(self):
        self.performance_log = []
        
    def monitor_performance(self, func: Callable) -> Callable:
        """Decorator to monitor function performance"""
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            start_time = time.time()
            start_memory = self._get_memory_usage()
            
            try:
                result = func(*args, **kwargs)
                
                execution_time = time.time() - start_time
                memory_used = self._get_memory_usage() - start_memory
                
                self._log_metrics(func.__name__, execution_time, memory_used)
                
                return result
                
            except Exception as e:
                logger.error(f"Error in {func.__name__}: {str(e)}")
                raise
                
        return wrapper
        
    def _get_memory_usage(self) -> float:
        """Get current memory usage"""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024  # MB
        except ImportError:
            return 0.0
            
    def _log_metrics(self, func_name: str, exec_time: float, memory_mb: float) -> None:
        """Log performance metrics"""
        metrics = {
            'function': func_name,
            'execution_time': f"{exec_time:.2f}s",
            'memory_used': f"{memory_mb:.2f}MB",
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        self.performance_log.append(metrics)
        logger.info(f"Performance metrics: {metrics}")
