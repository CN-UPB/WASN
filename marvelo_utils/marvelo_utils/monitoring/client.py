import base64
import time
from threading import Thread

import Pyro4
import numpy as np
from tornado import gen


class MonitoringClient:
    def __init__(self, host_ip, server_name):
        """
        Client to continuously read data from the monitoring server.

        Args:
            host_ip: IP of the device on which the monitoring server is running
            server_name: Name of the monitoring server
        """
        name_server = Pyro4.locateNS(host=host_ip)
        self.monitoring_server = Pyro4.Proxy(name_server.lookup(server_name))
        self.block_shape = self.monitoring_server.get_block_shape()
        self.dtype = self.monitoring_server.get_dtype()
        self.figures = []

    def start(self, doc):
        """
        Start worker thread self._worker.

        Args:
            doc: bokeh document
        """
        Thread(target=self._worker, args=(doc,), daemon=True).start()

    def _worker(self, doc):
        """
        worker thread, which is reading the remote data and updates
        self.figures in a callback of a bokeh document.

        Args:
            doc: bokeh document
        """
        block_idx = 0
        while True:
            while self.monitoring_server.get_block_id() == block_idx:
                # Wait until the data stored by the monitoring server is updated
                # if the currently stored data was already read
                time.sleep(.01)
            # Read the data form the monitoring server
            block_idx = self.monitoring_server.get_block_id()
            data = self.monitoring_server.get_data()['data']
            data = np.fromstring(
                base64.b64decode(data), dtype=self.dtype
            ).reshape(self.block_shape)
            for fig in self.figures:
                fig.update_data(data)
            doc.add_next_tick_callback(self.update_figures)

    @gen.coroutine
    def update_figures(self):
        """
        update sources of all registered figures for visualization.
        See also marvelo_utils.monitoring.visualization.Figure.
        """
        for fig in self.figures:
            fig.update_sources()
