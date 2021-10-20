"""
Read audio from a pipe and write it to a file (multiple audio channels
are supported)
"""
import argparse
import sys
import numpy as np
from scipy.io.wavfile import write
from utils import PipeReader


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='arguments')
    parser.add_argument("--inputs", "-i", action="append",
                        help='IDs of the input pipes of this module')
    parser.add_argument(
        "--channels_in", "-ci", type=int, metavar='CHANNELS_IN',
        help='Number of audio channels read out from the device'
    )
    parser.add_argument(
        "--channels_out", "-co", type=int, metavar='CHANNELS_OUT',
        help='Number of audio channels which should be written to the file'
    )
    parser.add_argument(
        "--block_len", "-b", type=int,
        help='length of the blocks to be read from the input pipe'
    )
    parser.add_argument(
        "--file_path", "-f", help='Defines where the wave file sould be stored'
    )
    parser.add_argument(
        "--audio_length", "-l", type=int, default=320000,
        help='Length in samples of the audio which should be written to a file'
    )
    parser.add_argument("--sampling_rate", "-sr", default=16000,
                        help='Sampling rate of the audio read form the device')
    args = parser.parse_args()
    if args.channels_out > args.channels_in:
        parser.error('argument CHANNELS_OUT must be <= argument CHANNELS_IN')

    reader = PipeReader(
        int(args.inputs[0]), (int(args.block_len), int(args.channels_in))
    )
    audio = np.zeros((args.audio_length, args.channels_out), dtype=np.float32)

    try:
        print('Start reading from device')
        sys.stdout.flush()
        # Read block by block from the input pipe until the desired amount of
        # audio samples is gathered
        block_cnt = 0
        while block_cnt < np.floor(args.audio_length / args.block_len):
            new_data = reader.read_block()
            start_block = block_cnt * args.block_len
            audio[start_block:start_block+args.block_len] = \
                new_data[:, :args.channels_out]
            block_cnt += 1
        new_data = reader.read_block()
        len_rest = args.audio_length - (block_cnt - 1) * args.block_len
        audio[-len_rest:] = new_data[:len_rest, :args.channels_out]
        if args.channels_out == 1:
            # Remove channel dimension if only one channel should be
            # written to the file
            audio = audio.squeeze(-1)

        # Write audio to the file
        write(args.file_path, args.sampling_rate, audio)
        print(f'Wrote audio to {args.file_path}')
        sys.stdout.flush()
    except Exception:
        # If an error occurs display the error in the console
        import traceback
        traceback.print_exc(file=sys.stdout)
        sys.stdout.flush()
