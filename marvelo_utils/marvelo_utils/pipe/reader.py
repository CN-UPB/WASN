import os
import numpy as np
from queue import Queue
from threading import Thread


class PipeReader:
    def __init__(self, fd, block_shape, dtype=np.float32):
        """
        Continuously reads pipe and saves blocks (with a certain block shape)
        in a queue. You need to call pipe_reader.start() to start the reader
        thread. By reading the pipe in a separate thread, a data loss due to
        a pipe overflow is prevented.

        Args:
            fd: file descriptor
            block_shape: the shape of the blocks to be read from the pipe
            dtype: the dtype of the data read from the pipe
        """
        self.fd = fd
        self.block_shape = block_shape
        self.dtype = dtype
        self.bytes_per_block = np.prod(block_shape) * dtype().nbytes
        self.queue = Queue()
        self.running = False

    def start(self):
        """
        Start the reader thread
        """
        Thread(target=self._worker, daemon=True).start()
        self.running = True

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

    def get_next_block(self):
        """
        get the next block from the pipe. Function call is blocking as long as
        the queue of blocks is empty.
        """
        assert self.running, 'pipe reader has not been started yet'
        return self.queue.get()

    def get_latest_block(self):
        """
        get the latest block from the pipe. If there are more than one block in
        the queue, the latest is returned and older ones are discarded.
        Function call is blocking as long as the queue of blocks is empty.
        """
        assert self.running, 'pipe reader has not been started yet'
        block = self.queue.get()
        while not self.queue.empty():
            block = self.queue.get()
        return block
