# Compressor Interface
# Only compressors extending this class will be compatible with the system. This ensures that all compressors have a consistent API for compressing embeddings.
# The `compress_embedding` method must be implemented by any subclass, and it should take an embedding as input and return a compressed version of that embedding. This allows the system to use different compression techniques without needing to change the way embeddings are handled elsewhere in the code.

class CompressorInterface:
    def __init__(self):
        pass

    def compress_embedding(self, embedding):
        raise NotImplementedError("compress_embedding method must be implemented by subclass")