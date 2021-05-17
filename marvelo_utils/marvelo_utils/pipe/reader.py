import os
import numpy as np
from queue import Queue
from threading import Thread


class PipeReader:
    def __init__(self, fd, block_shape, dtype=np.float32):
        self.fd = fd
        self.block_shape = block_shape
        self.dtype = dtype
        self.bytes_per_block = np.prod(block_shape) * dtype().nbytes
        self.queue = Queue()

    def start(self):
        Thread(target=self._worker, daemon=True).start()

    def _worker(self):
        buffer = list()
        byte_count = 0
        while True:
            while byte_count < self.bytes_per_block:
                buffer.append(os.read(self.fd, self.bytes_per_block))
                byte_count += len(buffer[-1])
            bytes = b"".join(buffer)
            buffer = [bytes[self.bytes_per_block:]]
            byte_count = len(buffer[-1])
            x = np.fromstring(
                bytes[:self.bytes_per_block], dtype=self.dtype
            ).reshape(self.block_shape)
            # print(x.tobytes() == bytes[:self.bytes_per_block])
            self.queue.put(x)

    def get_latest_block(self):
        block = self.queue.get()
        while not self.queue.empty():
            block = self.queue.get()
        return block

    def get_next_block(self):
        return self.queue.get()
