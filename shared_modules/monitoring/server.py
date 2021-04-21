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


@Pyro4.expose
@Pyro4.behavior(instance_mode="single")
class MonitioringServer(object):
    """
    Server to send results to another device where they can be displayed
    """
    def __init__(self, num_channels, block_len):
        """
        Args:
            num_channels: Number of channels read from the pipe
            block_len: Length of one block read from the pipe
        """
        self.num_channels = num_channels
        self.block_len = block_len
        self.data = \
            np.zeros((block_len, num_channels), dtype=np.float32).tobytes()
        self.block_id = 0

    def update_data(self, new_data):
        """Overwrite the current data to be send by the server"""
        self.data = b64decode(new_data['data'])
        self.block_id += 1

    def get_channels(self):
        """Return the number of channels of one block"""
        return self.num_channels

    def get_block_len(self):
        """Return the length of one block"""
        return self.block_len

    def get_data(self):
        """Return the currently stored data"""
        return self.data

    def get_block_id(self):
        """Return the ID of the currently stored block"""
        return self.block_id


def run_daemon(ip, server_names, num_channels, block_len):
    """
    Run the monitoring servers and register them in the name server

    Args:
        ip: IP of the device on which the Pyro name should be hosted
        server_names: Names over which the monitoring servers can be reached
        num_channels: List of the amount of channels of the single results
            to be monitored
        block_len: List of the lengths of one block of the single results
            to be monitored
    """
    servers = {
        MonitioringServer(ch, bl): name
        for name, ch, bl in zip(server_names, num_channels, block_len)
    }
    print('Start monitoring servers (Broadcast servers)')
    sys.stdout.flush()
    Pyro4.Daemon.serveSimple(servers, host=ip, ns=True)


def run_name_server(ip):
    """
    Run a Pyro name server. This allows an easy handling of the monitoring
    servers (hey can simply be addressed by the name they were given).
    Additionally, the name server will be used to easily reach the monitoring
    servers form another device on the network.

    Args:
        ip: IP of the device on which the Pyro name should be hosted
    """
    startNSloop(ip, storage='memory')


if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser(description='arguments')
        parser.add_argument("--inputs", "-i", action="append",
                            help='IDs of the input pipes of this module')
        help_msg = ('Comma separated enumeration of the amount of '
                    'channels of the single results to be monitored')
        parser.add_argument("--channels", "-c", help=help_msg)
        help_msg = ('Comma separated enumeration of the lengths of one block'
                    'of the single results to be monitored')
        parser.add_argument("--block_len", "-b", help=help_msg)
        help_msg = ('Network interface over which the monitoring'
                    'servers can be reached')
        parser.add_argument("--iface", "-if", default="eth0",
                            type=str, help=help_msg)

        parser.add_argument(
            "--name", "-n", default='monitoring', type=str,
            help='Names over which the monitoring servers can be reached'
        )
        args = parser.parse_args()

        num_channels = [int(ch) for ch in args.channels.split(',')]
        block_len = [int(bl) for bl in args.block_len.split(',')]
        server_names = args.name.split(',')
        iface = args.iface
        readers = \
            [PipeReader(int(input_), (bl, ch))
             for input_, bl, ch in zip(args.inputs, block_len, num_channels)]

        # Get the IP of the device. This is needed to be able to reach the
        # monitoring servers from another device on the network.
        ip = ni.ifaddresses(iface)[ni.AF_INET][0]['addr']

        # Start a Pyro name server. The name server is needed to be able to
        # easily reach the monitoring servers from another device on the
        # network.
        thread = Thread(target=run_name_server, daemon=True, args=(ip,))
        thread.start()

        # Start the monitoring servers and register then in the name server
        thread = Thread(target=run_daemon, daemon=True,
                        args=(ip, server_names, num_channels, block_len))
        thread.start()

        # Wait until the Pyro server is started
        time.sleep(1)
        sys.stdout.flush()

        # Collect the running monitoring servers to be able to update the data
        # which is stored by them.
        name_server = Pyro4.locateNS()
        monitoring_servers = \
            [Pyro4.Proxy(name_server.lookup(name)) for name in server_names]

        while True:
            # Collect new data from the input pipes and handle the new data to
            # the monitoring servers.
            new_data = [reader.read_block() for reader in readers]
            for monitoring_server, data in zip(monitoring_servers, new_data):
                monitoring_server.update_data(data.tobytes())
    except Exception:
        # If an error occurs display the error in the console
        import traceback
        traceback.print_exc(file=sys.stdout)
        sys.stdout.flush()
