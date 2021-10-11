"""
Live visualization of the results of the online weighted averaged (WACD)
sampling rate offset (SRO) estimation demo
"""
import argparse
from bokeh.layouts import column, row
from bokeh.server.server import Server
from math import ceil
import numpy as np
import scipy.signal as sig
from marvelo_utils.monitoring.client import MonitoringClient
from marvelo_utils.monitoring.visualization import MultiLine


class OnlineWACDVisualization(MultiLine):
    def __init__(self, monitoring_client, buffer_size,
                 scaling=1., win_range_smoothing=5, **kwargs):
        """
        Live visualization of the delays and SRO calculated by the WACD demo

        Args:
            monitoring_client: Client to read from the monitoring server
            buffer_size: Size of the buffer needed to store the values
                to be displayed
            win_range_smoothing: Length of the data windows used to reject
                outliers when calculation the updated range of the y-axis
        """
        super().__init__(monitoring_client, buffer_size, **kwargs)
        self.win_range_smoothing = win_range_smoothing
        self.scaling = scaling

    def update_figure(self):
        """Update the data to be displayed and the range of the y-axes"""
        # Update the displayed data
        y = [yi * self.scaling for yi in self.data.T]
        for src, yi in zip(self.sources, y):
            src.data['y'] = yi

        # Update the range of the y-axes
        # Utilize the monotonicity of the results (start and end of the
        # range are defined by the values at the start and the end of the
        # time intervall to be displayed)
        buffer_smoothed = sig.medfilt(
            self.data[:, 0] * self.scaling, self.win_range_smoothing
        )
        val_start = np.min(buffer_smoothed)
        val_end = np.max(buffer_smoothed)
        pad_width = np.maximum(1, 0.2 * abs(val_end - val_start))
        self.figure.y_range.update(
            start=val_start - pad_width, end=val_end + pad_width
        )


def modify_doc(doc):
    doc.add_root(
        column(sro_line_plot.figure, delay_line_plot.figure)
    )
    doc.title = "Synchronization"
    sro_client.start(doc)
    delay_client.start(doc)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='arguments')
    parser.add_argument(
        "--win_len", "-w", type=int, default=60,
        help='Length (in seconds) of the time window to be displayed'
    )
    parser.add_argument("--plot_width", "-pw", type=int, default=1500,
                        help='Width (in pixels) of the plot')
    parser.add_argument("--plot_height", "-ph", type=int, default=425,
                        help='Height (in pixels) of the plot')
    parser.add_argument("--font_size", "-f", type=int, default=30,
                        help='Font size used in the plot')
    parser.add_argument("--server_name", "-n", default='monitoring_sync_demo',
                        help='Name of the monitoring server')
    msg = 'IP of the device on which the monitoring server is running'
    parser.add_argument("--host_ip", "-ip", help=msg)
    args = parser.parse_args()

    data_rate = 16000 / 2048
    buffer_size = ceil(args.win_len * data_rate)
    x_ticks = np.linspace(0, args.win_len, buffer_size, dtype=np.float32)

    sro_client = MonitoringClient(args.host_ip, 'sro')
    sro_line_plot = OnlineWACDVisualization(
        sro_client,
        buffer_size=buffer_size,
        title='Sampling Rate Offset (SRO) estimation in parts-per-millions (ppm)',
        xlabel='time / s',
        ylabel='SRO / ppm',
        x_ticks=x_ticks,
        y_range=(-10, 10),
        plot_width=args.plot_width,
        plot_height=args.plot_height,
        font_size=args.font_size,
        labels=None,
        scaling=1e6
    )

    delay_client = MonitoringClient(args.host_ip, 'delay')
    delay_line_plot = OnlineWACDVisualization(
        delay_client,
        buffer_size=buffer_size,
        title='Accumulating Time Drift (ATD) estimation in samples (smp)',
        xlabel='time / s',
        ylabel='ATD / samples',
        x_ticks=x_ticks,
        y_range=(-10, 10),
        plot_width=args.plot_width,
        plot_height=args.plot_height,
        font_size=args.font_size,
        labels=None
    )

    server = Server({'/': modify_doc})
    server.start()
    server.io_loop.add_callback(server.show, "/")
    server.io_loop.start()
