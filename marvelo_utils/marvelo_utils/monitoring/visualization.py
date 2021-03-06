import numpy as np
from bokeh.models import ColumnDataSource, Legend
from bokeh.models.mappers import LinearColorMapper
from bokeh.palettes import Category10
from bokeh.plotting import figure


class Figure:
    def __init__(self, monitoring_client, buffer_size=1, **kwargs):
        """
        Base class to implement figures which are connected with a monitoring
        client and automatically updated whenever the monitoring client
        receives a new data block. Child classes only need to implement
        create_figure and update_sources methods. kwargs of the __init__ are
        forwarded to create_figure.
        """
        self.monitoring_client = monitoring_client
        self.buffer_size = buffer_size
        data_shape = [*monitoring_client.block_shape]
        data_shape[0] *= buffer_size
        self.data = np.zeros(data_shape, dtype=monitoring_client.dtype)
        self.figure = self.create_figure(**kwargs)
        monitoring_client.figures.append(self)

    def update_data(self, data):
        """
        updates data buffer. Called by monitoring client whenever new data is
        received.
        """
        block_len, *_ = data.shape
        self.data = np.roll(self.data, -block_len, axis=0)
        self.data[-block_len:] = data

    def create_figure(self, **kwargs):
        raise NotImplementedError

    def update_figure(self):
        raise NotImplementedError


class VerticalBars(Figure):
    def create_figure(
            self,
            title='',
            xlabel='',
            ylabel='',
            plot_width=750,
            plot_height=500,
            font_size=30,
            y_range=(0., 1.),
            labels=None,
    ):
        """
        creating a bokeh vertical bars plot.

        Args:
            title:
            xlabel: label for x axis
            ylabel: label for y axis
            plot_width:
            plot_height:
            font_size:
            y_range: tuple (ymin, ymax) of minimal y value (ymin)
                and maximal y value (ymax)
            labels: Optional list of labels for the vertical bars.
        """
        assert self.buffer_size == 1, self.buffer_size
        assert self.data.ndim == 2, self.data.shape
        num_classes = self.data.shape[-1]
        if labels is None:
            labels = [str(i) for i in range(num_classes)]
        assert len(labels) == num_classes, (num_classes, len(labels), labels)
        self.source = ColumnDataSource(
            data=dict(
                labels=labels,
                y=np.zeros(num_classes).tolist(),
                color=(Category10[10])[:num_classes]
            )
        )
        fig = figure(
            x_range=labels,
            plot_height=plot_height,
            plot_width=plot_width,
            toolbar_location=None,
            title=title
        )
        fig.vbar(
            x='labels',
            top='y',
            width=.75,
            source=self.source,
            line_color='white',
            fill_color='color'
        )
        fig.y_range.start, fig.y_range.end = y_range
        fig.xaxis.axis_label = xlabel
        fig.yaxis.axis_label = ylabel
        fig.title.text_font_size = f'{font_size}pt'
        fig.axis.major_label_text_font_size = f'{font_size}pt'
        fig.xaxis.axis_label_text_font_size = f'{font_size}pt'
        fig.yaxis.axis_label_text_font_size = f'{font_size}pt'
        return fig

    def update_figure(self):
        """
        updating the vertical bars' heights
        """
        self.source.data['y'] = self.data[-1].tolist()


class MultiLine(Figure):
    def create_figure(
            self,
            title='',
            xlabel='',
            ylabel='',
            x_ticks=None,
            plot_width=750,
            plot_height=500,
            font_size=30,
            x_range=None,
            y_range=(0, 1),
            labels=None,
            line_width=5,
    ):
        """
        Creating a bokeh multi line plot.

        Args:
            title: title of the plot
            xlabel: label for x axis
            ylabel: label for y axis
            x_ticks: ticks of the x-axis. If None, [0, 1, 2, ...] will be used.
            plot_width: width (in pixels) of the single plot
            plot_height: height (in pixels) of the single plot
            font_size: font size used in the plot
            y_range: tuple (ymin, ymax) of minimal y value (ymin)
                and maximal y value (ymax)
            labels: optional list of labels for the lines
            line_width: width of the lines
        """
        assert self.data.ndim == 2, self.data.shape
        num_time_steps, num_lines = self.data.shape
        if labels is None and num_lines > 1:
            labels = [str(i) for i in range(num_lines)]
            assert len(labels) == num_lines, (num_lines, len(labels), labels)

        if x_ticks is None:
            x_ticks = np.arange(num_time_steps).astype(np.float32)
        self.sources = [
            ColumnDataSource(
                data=dict(
                    x=x_ticks,
                    y=np.zeros(num_time_steps).astype(np.float32)
                )
            ) for _ in range(num_lines)
        ]

        fig = figure(
            plot_height=plot_height,
            plot_width=plot_width,
            title=title,
            toolbar_location=None,
            x_range=x_range,
            y_range=y_range,
        )
        items = []
        for line_idx in range(num_lines):
            p = fig.line(
                'x', 'y',
                source=self.sources[line_idx],
                line_color=(Category10[10])[line_idx],
                line_width=line_width
            )
            if labels is not None:
                items.append((labels[line_idx], [p]))
        if labels is not None:
            legend = Legend(
                items=items,
                location='center',
                glyph_height=50,
                glyph_width=30,
            )
            fig.add_layout(legend, 'right')
        fig.xaxis.axis_label = xlabel
        fig.yaxis.axis_label = ylabel
        fig.title.text_font_size = f'{font_size}pt'
        fig.axis.major_label_text_font_size = f'{font_size}pt'
        if labels is not None:
            fig.legend.label_text_font_size = f'{font_size}pt'
        fig.xaxis.axis_label_text_font_size = f'{font_size}pt'
        fig.yaxis.axis_label_text_font_size = f'{font_size}pt'
        if x_range is None:
            fig.x_range.range_padding = 0
        #fig.y_range.start, fig.y_range.end = y_range
        return fig

    def update_figure(self):
        y = [yi for yi in self.data.T]
        for src, yi in zip(self.sources, y):
            src.data['y'] = yi


class Image(Figure):
    def create_figure(
            self,
            title='',
            xlabel='',
            ylabel='',
            plot_width=750,
            plot_height=500,
            font_size=30,
            low=-3, high=3,
    ):
        """
        creating a bokeh image where data values serve as pixel intensity.

        Args:
            title:
            xlabel: label for x axis
            ylabel: label for y axis
            plot_width:
            plot_height:
            font_size:
            low: lowest intensity. values below are clamped.
            high: highest intensity. values above are clamped.
        """
        assert self.data.ndim == 2, self.data.shape
        num_frames, num_bins = self.data.shape
        x = np.arange(num_frames)
        x = np.repeat(x, num_bins, axis=0)
        y = np.repeat(np.arange(num_bins)[None], num_frames, axis=0).flatten()
        v = np.zeros_like(y).tolist()

        self.source = ColumnDataSource(
            data=dict(x=x, y=y, v=v)
        )
        fig = figure(
            plot_height=plot_height,
            plot_width=plot_width,
            toolbar_location=None,
            title=title
        )
        mapper = LinearColorMapper(palette='Viridis256', low=low, high=high)
        fig.rect(
            'x',
            'y',
            color={'field': 'v', 'transform': mapper},
            width=1,
            height=1,
            source=self.source
        )
        fig.x_range.range_padding = 0
        fig.y_range.range_padding = 0
        fig.xaxis.axis_label = xlabel
        fig.yaxis.axis_label = ylabel
        fig.title.text_font_size = f'{font_size}pt'
        fig.axis.major_label_text_font_size = f'{font_size}pt'
        fig.xaxis.axis_label_text_font_size = f'{font_size}pt'
        fig.yaxis.axis_label_text_font_size = f'{font_size}pt'
        return fig

    def update_figure(self):
        self.source.data['v'] = self.data.flatten().tolist()
