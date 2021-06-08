"""
Module to read audio from a sound card and write it to a pipe to another module

To get the hardware_interface for the desired device run the following command:
    python read_from_device.py -l
By this a list of available audio devices is shown in the console. The hardware
interface to be used to read from the device corresponds to the integer number
shown in the first column.
"""
import argparse
import sys
import numpy as np
from queue import Queue
import sounddevice as sd
from time import sleep
from marvelo_utils.pipe.writer import BufferingPipeWriter


class AudioReader:
    def __init__(self, hw_interface, n_chs, sampling_rate, start_delay):
        """
        Args:
            hw_interface: Hardware Interface of the device to be used
            n_chs: Number of channel to be read
            sampling_rate: Target sampling rate
            start_delay: Number of frames to drop after start
        """
        self.queue = Queue()
        self.start_delay = start_delay
        # Start a stream to read the data frame by frame from the device
        self.audio_stream = sd.InputStream(
            device=hw_interface, channels=n_chs, samplerate=sampling_rate,
            callback=self.read_frame
        )
        self.audio_stream.start()

    def read_frame(self, indata, *args):
        """
        Read current frame from the device and write it to the output pipe.
        This is called (from a separate thread) for each audio frame.
        """
        if self.start_delay > 0:
            # Discard frame during start up phase
            if self.start_delay < len(indata):
                rest = len(indata) - self.start_delay
                self.queue.put(indata[-rest:])
            self.start_delay -= len(indata)
        else:
            self.queue.put(indata)


if __name__ == '__main__':
    try:
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument(
            '-l', '--list-devices', action='store_true',
            help='Show list of audio devices and exit')
        args, remaining = parser.parse_known_args()
        if args.list_devices:
            # If available audio devices should be listed display them in the
            # console and exit the module
            print(sd.query_devices())
            parser.exit(0)

        parser = argparse.ArgumentParser()
        parser.add_argument('-c', '--channels', type=int,
                            metavar='CHANNELS', help='Number of channels')
        parser.add_argument(
            '-f', '--frame_size', type=int, default=128, metavar='FRAME_SIZE',
            help='Number of samples per channel in one frame'
        )
        parser.add_argument(
            '-s', '--sampling_rate', type=int, default=16000,
            metavar='SAMPLING_RATE', help='Target sampling rate'
        )
        parser.add_argument(
            '-sd', '--start_delay', type=int, default=0,
            metavar='START_DELAY', help='Number of frames to drop after start.'
        )
        parser.add_argument(
            '-i', '--hardware_interface', type=int,
            metavar='HARDWARE_INTERFACE',
            help='Hardware Interface of the to be used device'
        )
        parser.add_argument(
            '-b', '--bits_per_sample', type=int, default=32,
            choices=[16, 32, 64], metavar='BITS_PER_SAMPLE',
            help='Amount of bits to be used to represent an audio sample'
        )
        parser.add_argument(
            "--outputs", "-o", action="append", type=int,
            help='List of pipes to which the audio should be written.'
        )
        args = parser.parse_args(remaining)
        if args.hardware_interface is None:
            parser.error('You have to specify the device.')
        if (args.hardware_interface >= len(sd.query_devices())
                or args.hardware_interface < 0):
            msg = (f'argument HARDWARE_INTERFACE: must be '
                   f'>= 0 and <= {len(sd.query_devices())}')
            parser.error(msg)
        max_ch = \
            sd.query_devices()[args.hardware_interface]['max_input_channels']
        if args.channels < 1 or args.channels > max_ch:
            parser.error(f'argument CHANNELS: must be >= 1 and <= {max_ch}')
        if args.frame_size < 1:
            parser.error('argument FRAME_SIZE: must be >= 1')
        if args.sampling_rate < 1:
            parser.error('argument SAMPLING_RATE: must be >= 1')
        if args.start_delay < 0:
            parser.error('argument START_DELAY: must be >= 0')

        if args.bits_per_sample == 16:
            d_type = np.float16
        elif args.bits_per_sample == 32:
            d_type = np.float32
        elif args.bits_per_sample == 64:
            d_type = np.float64

        # Read continuously from the device and write the audio
        # to the output pipes
        start_delay = args.start_delay * args.frame_size
        audio_reader = AudioReader(args.hardware_interface, args.channels,
                                   args.sampling_rate, args.start_delay)
        pipe_writer = BufferingPipeWriter(
            audio_reader.queue, args.outputs, (args.frame_size, ), None, d_type
        )
        pipe_writer.start()

        # Keep the main thread alive as long as the audio stream is available
        while audio_reader.audio_stream.active:
            sleep(.1)
    except Exception:
        # If an error occurs display the error in the console
        import traceback
        traceback.print_exc(file=sys.stdout)
        sys.stdout.flush()
