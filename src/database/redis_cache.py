import redis
import json
import logging
from typing import Any, Optional, Union

logger = logging.getLogger(__name__)

class RedisCache:
    def __init__(self, host='localhost', port=6379, db=0, password=None, socket_timeout=30):
        self.host = host
        self.port = port
        self.db = db
        self.password = password
        self.socket_timeout = socket_timeout
        self.redis_client = None
        self.is_connected = False

    def connect(self):
        """Establish Redis connection"""
        try:
            self.redis_client = redis.StrictRedis(
                host=self.host,
                port=self.port,
                db=self.db,
                password=self.password,
                decode_responses=True,
                socket_timeout=self.socket_timeout,
                socket_connect_timeout=self.socket_timeout,
                retry_on_timeout=True
            )
            
            # Test connection
            self.redis_client.ping()
            self.is_connected = True
            logger.info("Connected to Redis successfully")
            return True
            
        except redis.ConnectionError as e:
            logger.error(f"Could not connect to Redis: {e}")
            self.is_connected = False
            return False
        except Exception as e:
            logger.error(f"Redis connection error: {e}")
            self.is_connected = False
            return False

    def disconnect(self):
        """Close Redis connection"""
        if self.redis_client:
            self.redis_client.close()
            self.is_connected = False
            logger.info("Disconnected from Redis")

    def _ensure_connection(self):
        """Ensure Redis connection is active"""
        if not self.is_connected:
            return self.connect()
        
        try:
            self.redis_client.ping()
            return True
        except:
            return self.connect()

    def set(self, key: str, value: Any, ex: Optional[int] = None) -> bool:
        """Set a key-value pair with optional expiration"""
        try:
            if not self._ensure_connection():
                return False
            
            # Serialize complex objects to JSON
            if isinstance(value, (dict, list)):
                value = json.dumps(value)
            
            self.redis_client.set(key, value, ex=ex)
            logger.debug(f"Set key {key} in Redis")
            return True
            
        except Exception as e:
            logger.error(f"Error setting key {key} in Redis: {e}")
            return False

    def get(self, key: str) -> Optional[Any]:
        """Get value by key"""
        try:
            if not self._ensure_connection():
                return None
            
            value = self.redis_client.get(key)
            if value is None:
                return None
            
            # Try to deserialize JSON
            try:
                return json.loads(value)
            except (json.JSONDecodeError, TypeError):
                return value
                
        except Exception as e:
            logger.error(f"Error getting key {key} from Redis: {e}")
            return None

    def delete(self, key: str) -> bool:
        """Delete a key"""
        try:
            if not self._ensure_connection():
                return False
            
            result = self.redis_client.delete(key)
            logger.debug(f"Deleted key {key} from Redis")
            return bool(result)
            
        except Exception as e:
            logger.error(f"Error deleting key {key} from Redis: {e}")
            return False

    def exists(self, key: str) -> bool:
        """Check if key exists"""
        try:
            if not self._ensure_connection():
                return False
            
            return bool(self.redis_client.exists(key))
            
        except Exception as e:
            logger.error(f"Error checking existence of key {key} in Redis: {e}")
            return False

    def expire(self, key: str, seconds: int) -> bool:
        """Set expiration for a key"""
        try:
            if not self._ensure_connection():
                return False
            
            result = self.redis_client.expire(key, seconds)
            return bool(result)
            
        except Exception as e:
            logger.error(f"Error setting expiration for key {key} in Redis: {e}")
            return False

    def get_stats(self) -> dict:
        """Get Redis statistics"""
        try:
            if not self._ensure_connection():
                return {}
            
            info = self.redis_client.info()
            return {
                'connected_clients': info.get('connected_clients', 0),
                'used_memory_human': info.get('used_memory_human', '0B'),
                'keyspace_hits': info.get('keyspace_hits', 0),
                'keyspace_misses': info.get('keyspace_misses', 0)
            }
            
        except Exception as e:
            logger.error(f"Error getting Redis stats: {e}")
            return {}

    def health_check(self) -> bool:
        """Check Redis connectivity"""
        try:
            if not self._ensure_connection():
                return False
            
            self.redis_client.ping()
            return True
            
        except Exception as e:
            logger.error(f"Redis health check failed: {e}")
            return False

    def cache_fraud_result(self, transaction_id: str, result: dict, expiration: int = 3600):
        """Cache fraud detection result"""
        key = f"fraud_result:{transaction_id}"
        return self.set(key, result, ex=expiration)

    def get_fraud_result(self, transaction_id: str) -> Optional[dict]:
        """Get cached fraud detection result"""
        key = f"fraud_result:{transaction_id}"
        return self.get(key)