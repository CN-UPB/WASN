"""
Resample the given signal using the given sampling rate offset (SRO) as
described in Sec. 5 of "Efficient Sampling Rate Offset Compensation - An
Overlap-Save Based Approach". In it's current form the resampler is designed to
simulate the given SRO. If the resampling should be used to compensate a given
SRO of x ppm sro=-x has to be passed to the resampler.
"""
import numpy as np
from scipy.signal.windows import hann


class STFTResampler:
    def __init__(self, sro=None, fft_size=8192, buffer_size=64000, buffer_shift=8192):
        """
        Args:
            sro: If sro is not None sro defines the SRO in ppm to be simulated
            fft_size: FFT-size used by the resampler. The block-size
                corresponds to fft_size / 2 and the block-shift to fft_size/4.
            buffer_size: Length of the buffer (needed due to the changed amount
                of samples caused by the SRO) to store the signal
            buffer_shift: Shift of the buffer if the delay caused by the SRO if
                the buffer becomes too empty or too full
        """
        self.sro = sro
        if self.sro is not None:
            self.sro /= 1e6
        self.update_overall_delay = True
        self.fft_size = fft_size
        self.block_len = int(fft_size // 2)
        self.win = hann(self.block_len, sym=False)
        self.k = np.fft.fftshift(np.arange(-fft_size / 2, fft_size / 2))
        self.buffer_size = buffer_size
        self.input_buffer = np.zeros(self.buffer_size + 2 * self.block_len)
        self.n_zero = len(self.input_buffer) // 2
        self.len_buffered = 0
        self.buffer_shift = buffer_shift
        self.output_buffer = np.zeros(self.block_len)
        self.overall_delay = 0

    def __call__(self, sig, sro=None):
        """
        Resamples the input signal
        Args:
            sig: One frame of the signal to be resampled
            sro: SRO used to update the current delay caused by the SRO
                over time (will be ignored of self.sro is not None)
        """
        # Write new frame to the buffer
        start = self.n_zero + self.len_buffered
        self.input_buffer[start:start+len(sig)] = sig
        self.len_buffered += len(sig)

        # Update the overall delay caused by the SRO over time
        if self.update_overall_delay:
            if self.sro is not None:
                # Use the constant SRO to be simulated if it is given to
                # update the overall delay
                self.overall_delay += self.sro * self.block_len / 2
            else:
                # Use the SRO hand over to the function to update the
                # overall delay
                self.overall_delay += sro * self.block_len / 2
            self.update_overall_delay = False

        # Split the overall delay into an integer part and a rest delay
        integer_delay = np.round(self.overall_delay)
        rest_delay = integer_delay - self.overall_delay

        # Produce the next block of the resampled signal if the enough values
        # are stored in the buffer
        if self.len_buffered >= self.block_len - integer_delay:
            # Produce the integer delay using the signal buffer
            start = int(self.n_zero - integer_delay)
            # Shift the buffer if the delay caused by the SRO gets too
            # large or too small(buffer becomes too empty or too full)
            if (start + self.block_len
                    > len(self.input_buffer) - self.buffer_shift):
                self.input_buffer = \
                    np.roll(self.input_buffer, -self.buffer_shift)
                self.n_zero -= self.buffer_shift
            if start < self.buffer_shift:
                self.input_buffer = \
                    np.roll(self.input_buffer, self.buffer_shift)
                self.n_zero += self.buffer_shift
            start = int(self.n_zero - integer_delay)
            block = self.input_buffer[start:start+self.block_len]

            # Use a phase shift to produce the non-integer delay
            phase = \
                np.exp(2 * np.pi * 1j * self.k / self.fft_size * rest_delay)
            sig_fft = np.fft.fft(self.win * block, self.fft_size)

            # Update input and output buffer
            self.output_buffer = \
                np.roll(self.output_buffer, -int(self.block_len//2))
            self.output_buffer[int(self.block_len // 2):] = \
                np.zeros(int(self.block_len // 2))
            self.output_buffer += \
                np.real(np.fft.ifft(sig_fft * phase))[:self.block_len]
            self.input_buffer = \
                np.roll(self.input_buffer, -int(self.block_len//2))
            self.input_buffer[-int(self.block_len//2):] = \
                np.zeros(int(self.block_len//2))
            self.len_buffered -= int(self.block_len // 2)
            self.update_overall_delay = True
            return self.output_buffer[:int(self.block_len//2)]
