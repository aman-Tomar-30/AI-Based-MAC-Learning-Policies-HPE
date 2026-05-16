class MockRedis:
    """Simulates Redis functionality using a Python dictionary."""
    def __init__(self):
        self.storage = {}

    def set(self, key, value):
        self.storage[key] = value

    def get(self, key):
        return self.storage.get(key)

redis_client = MockRedis()