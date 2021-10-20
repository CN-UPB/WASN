"""
Use STFT-resampling to simulate an SRO
"""
import argparse
import sys
from resample import STFTResampler
from marvelo_utils.pipe.reader import PipeReader
from marvelo_utils.pipe.writer import PipeWriter


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='arguments')
    parser.add_argument("--inputs", "-i", action="append",
                        help='IDs of the input pipes of this module')
    parser.add_argument("--sro", "-s", type=int)
    parser.add_argument(
        "--block_len", "-b", type=int, default=128,
        help='Length of the blocks to be read from the input pipe'
    )
    parser.add_argument("--outputs", "-o", action="append", type=int)
    args = parser.parse_args()

    resampler = STFTResampler(args.sro)
    reader = PipeReader(int(args.inputs[0]), int(args.block_len))
    writer = PipeWriter(args.outputs)

    try:
        reader.start()
        while True:
            frame = reader.get_next_block()

            # Resample signal
            output_block = resampler(frame)

            # If enough audio samples are stored in the internal buffer of the
            # resampler the buffered audio is processed a the resampled audio
            # (FFT-size / 4 samples) is returned. Otherwise None will
            # be returned.
            if output_block is not None:
                writer.write_next_block(output_block)
    except:
        # If an error occurs display the error in the console
        import traceback
        traceback.print_exc(file=sys.stdout)
        sys.stdout.flush()
