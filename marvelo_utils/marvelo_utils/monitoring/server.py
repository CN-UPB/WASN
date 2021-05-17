"""
Module to send all results to another device on the network where they
can be displayed
"""
from __future__ import print_function
import argparse
from base64 import b64decode
import sys
import time
import numpy as np
import netifaces as ni
import Pyro4
from Pyro4.naming import startNSloop
from threading import Thread
from utils import PipeReader
import json


@Pyro4.expose
@Pyro4.behavior(instance_mode="single")
class MonitioringServer(object):
    """
    Server to send results to another device where they can be displayed
    """
    def __init__(self, block_shape, dtype='float32'):
        """
        Args:
            block_shape: shape of a block as tuple or list
            dtype: dtype of a block as string
        """
        self.block_shape = tuple(block_shape)
        self.dtype = dtype
        self.data = np.zeros(block_shape, dtype=dtype).tobytes()
        self.block_id = 0

    def update_data(self, new_data):
        """Overwrite the current data to be send by the server"""
        self.data = b64decode(new_data['data'])
        self.block_id += 1

    def get_block_shape(self):
        """Return the block shape of one block"""
        return self.block_shape

    def get_dtype(self):
        """Return the dtype of one block"""
        return self.dtype

    def get_data(self):
        """Return the currently stored data"""
        return self.data

    def get_block_id(self):
        """Return the ID of the currently stored block"""
        return self.block_id


def run_daemon(ip, server_names, block_shape, dtype):
    """
    Run the monitoring servers and register them in the name server

    Args:
        ip: IP of the device on which the Pyro name should be hosted
        server_names: Names over which the monitoring servers can be reached
        block_shape: List of the block shapes of the single results
            to be monitored
        dtype: List of the block dtypes of the single results
            to be monitored
    """
    servers = {
        MonitioringServer(bs, dt): name
        for name, bs, dt in zip(server_names, block_shape, dtype)
    }
    print('Start monitoring servers (Broadcast servers)')
    sys.stdout.flush()
    Pyro4.Daemon.serveSimple(servers, host=ip, ns=True)


def run_name_server(ip):
    """
    Run a Pyro name server. This allows an easy handling of the monitoring
    servers (they can simply be addressed by the name they were given).
    Additionally, the name server will be used to easily reach the monitoring
    servers form another device on the network.

    Args:
        ip: IP of the device on which the Pyro name should be hosted
    """
    startNSloop(ip, storage='memory')


if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser(description='arguments')
        parser.add_argument(
            "--inputs", "-i", action="append",
            help='IDs of the input pipes of this module'
        )
        help_msg = (
            'json encoded list or list of lists stating the block shapes of '
            'the data on the input pipes as json array. Note that shapes have '
            'to be chosen such that blocks come with the same block rate.'
        )
        parser.add_argument("--block_shape", "-bs", help=help_msg)
        help_msg = (
            'dtypes of data from the input pipes separated by comma.'
            'Default is float32'
        )
        parser.add_argument("--dtype", "-dt", default=None, help=help_msg)
        help_msg = (
            'Network interface over which the monitoring servers can be'
            'reached. Default is eth0.'
        )
        parser.add_argument(
            "--iface", "-if", default="eth0", type=str, help=help_msg
        )
        help_msg = (
            'Names over which the monitoring servers can be reached.'
            'Default is "monitoring".'
        )
        parser.add_argument(
            "--name", "-n", default='monitoring', type=str, help=help_msg
        )
        args = parser.parse_args()

        server_names = args.name.split(',')
        dtype = args.dtype
        if dtype is None:
            dtype = len(server_names) * ['float32']
        else:
            dtype = dtype.split(',')
        assert len(server_names) == len(dtype), (server_names, dtype)
        iface = args.iface
        block_shape = json.loads(args.block_shape)
        assert isinstance(block_shape, list) and len(block_shape) > 0, block_shape
        if not isinstance(block_shape[0], list):
            block_shape = [block_shape]
        assert isinstance(block_shape[0][0], int), block_shape
        readers = [
            PipeReader(int(input_), bs, dt)
            for input_, bs, dt in zip(args.inputs, block_shape, dtype)
        ]
        [reader.start() for reader in readers]
        # Get the IP of the device. This is needed to be able to reach the
        # monitoring servers from another device on the network.
        ip = ni.ifaddresses(iface)[ni.AF_INET][0]['addr']

        # Start a Pyro name server. The name server is needed to be able to
        # easily reach the monitoring servers from another device on the
        # network.
        thread = Thread(target=run_name_server, daemon=True, args=(ip,))
        thread.start()

        # Start the monitoring servers and register then in the name server
        thread = Thread(
            target=run_daemon, daemon=True,
            args=(ip, server_names, block_shape, dtype)
        )
        thread.start()

        # Wait until the Pyro server is started
        time.sleep(1)
        sys.stdout.flush()

        # Collect the running monitoring servers to be able to update the data
        # which is stored by them.
        name_server = Pyro4.locateNS()
        monitoring_servers = [
            Pyro4.Proxy(name_server.lookup(name)) for name in server_names
        ]
        while True:
            # Collect new data from the input pipes and handle the new data to
            # the monitoring servers.
            for reader, monitoring_server in zip(readers, monitoring_servers):
                data = reader.get_next_block()
                monitoring_server.update_data(
                    data.astype(monitoring_server.get_dtype()).tobytes()
                )
    except Exception:
        # If an error occurs display the error in the console
        import traceback
        traceback.print_exc(file=sys.stdout)
        sys.stdout.flush()
