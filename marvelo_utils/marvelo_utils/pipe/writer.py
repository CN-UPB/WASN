import os
import numpy as np
from threading import Thread


class PipeWriter:
    def __init__(self, fds, dtype=np.float32):
        """
        Helper class to write data to the output pipes of a module

        Args:
            fds: file descriptors representing the pipes
            dtype: type of the data to be written to the pipes
        """
        self.pipes = [os.fdopen(int(fd), 'wb') for fd in fds]
        self.dtype = dtype

    def write_next_block(self, data):
        for pipe in self.pipes:
            pipe.write(data.astype(self.dtype))
            pipe.flush()


class BufferingPipeWriter:
    def __init__(self,
                 queue,
                 fds,
                 block_shape,
                 init_buffer=None,
                 dtype=np.float32):
        """
        Continuously reads blocks from a queue and write the read blocks to
        pipes. You need to call pipe_writer.start() to start the writer thread.

        Args:
            queue: queue containing the data to be written to pipes
            fds: file descriptors representing the pipes
            block_shape: shape of the blocks written to the pipes. The first
                axis of the blocks is interpreted as time dimension. If
                block_shape=None the data blocks from the queue will be written
                directly to the pipes. Otherwise, the data will be buffered
                until a block of shape block_shape is available.
            init_buffer: initially buffered data. If None an empty buffer
                will be created.
            dtype: type of the data to be written to the pipes
        """
        self.queue = queue
        self.pipes = [os.fdopen(int(fd), 'wb') for fd in fds]
        self.dtype = dtype
        self.block_shape = block_shape
        if init_buffer is not None:
            self.buffer = list(init_buffer)
        else:
            self.buffer = []

    def start(self):
        """
        Start the writer thread
        """
        Thread(target=self._worker, daemon=True).start()

    def _worker(self):
        while True:
            # Read new data from the queue.
            self.buffer += list(self.queue.get())

            # Get all complete data blocks and write them to the pipes
            for block in self._get_blocks():
                for pipe in self.pipes:
                    pipe.write(block.astype(self.dtype).tobytes())
                    pipe.flush()

    def _get_blocks(self):
        """
        Return all complete blocks and buffer the remaining data
        """
        n_blocks = len(self.buffer) // self.block_shape[0]
        blocks = [self.buffer[i*self.block_shape[0]:(i+1)*self.block_shape[0]]
                  for i in range(n_blocks)]
        self.buffer = self.buffer[n_blocks*self.block_shape[0]:]
        return np.asarray(blocks)
