"""

Implementation of the Double-Cross-Correlation Processor (DXCP) with phase
transform (PhaT) for a blind Sampling-Rate Offset (SRO) and Time-Offset (STO)
estimation from [1].

[1] Chinaev A., Thuene P., and Enzner G., 'Double-Cross-Correlation Processing
for Blind Sampling-Rate and Time-Offset Estimation', IEEE/ACM Trans. on Audio,
Speech, and Language Proc., vol. 29, pp. 1881-1896, 2021.

"""

#!/usr/bin/env python3.6

# region... Implementation of DXCP-PhaT-CL as a MARVELO-Block
import sys
import math
import numpy as np
from scipy import signal
# endregion


class DXCPPhaT:
    
    # default parameters (configuration) defined as class attributes
    defaultParams = {
        # reference sampling rate
        'RefSampRate_fs_Hz': 16000, 
        # frame size (power of 2) of input data
        'FrameSize': 2**11, 
        # frame shift of DXCP-PhaT (power of 2 & >= FrameSize)
        'FFTshift': 2**13, 
        # FFT size of DXCP-PhaT (power of 2 & >= FFTshift)
        'FFTsize': 2**15, 
        # accumulation time in sec (usually 5s as in DXCP)
        'AccumTime_B_sec': 5, 
        # resetting period of DXCP-PhaT in sec. Default: 30
        'ResetPeriod_sec': 30, 
        # smoothing constant for GCSD1 averaging (DXCP-PhaT)
        'SmoConst_CSDPhaT_alpha': .5, 
        # additional waiting for container filling (>InvShiftFactor-1)
        'AddContWait_NumFr': 0, 
        # settling time of CSD-2 averaging
        'SettlingCSD2avg_NumFr': 4, 
        # minimum value of |Z1*conj(Z2)| to avoid devision by 0 in GCC-PhaT
        'Z_12_abs_min': 1e-12, 
        # maximum absolute SRO value possible to estimate (-> Lambda)
        'SROmax_abs_ppm': 1000, 
        # flag for displaying estimated values in terminal
        'Flag_DisplayResults': 1
    }


    def __init__(
        self,
        RefSampRate_fs_Hz = defaultParams['RefSampRate_fs_Hz'], 
        FrameSize = defaultParams['FrameSize'], 
        FFTshift = defaultParams['FFTshift'], 
        FFTsize = defaultParams['FFTsize'], 
        AccumTime_B_sec = defaultParams['AccumTime_B_sec'], 
        ResetPeriod_sec = defaultParams['ResetPeriod_sec'], 
        SmoConst_CSDPhaT_alpha = defaultParams['SmoConst_CSDPhaT_alpha'], 
        AddContWait_NumFr = defaultParams['AddContWait_NumFr'], 
        SettlingCSD2avg_NumFr = defaultParams['SettlingCSD2avg_NumFr'], 
        Z_12_abs_min = defaultParams['Z_12_abs_min'], 
        SROmax_abs_ppm = defaultParams['SROmax_abs_ppm'], 
        Flag_DisplayResults = defaultParams['Flag_DisplayResults']
        ):
        
        # Main parameters (configuration)
        self.RefSampRate_fs_Hz = RefSampRate_fs_Hz
        self.FrameSize = FrameSize
        self.FFTshift = FFTshift
        self.FFTsize = FFTsize
        self.AccumTime_B_sec = AccumTime_B_sec
        self.ResetPeriod_sec = ResetPeriod_sec
        self.SmoConst_CSDPhaT_alpha = SmoConst_CSDPhaT_alpha
        self.AddContWait_NumFr = AddContWait_NumFr
        self.SettlingCSD2avg_NumFr = SettlingCSD2avg_NumFr
        self.Z_12_abs_min = Z_12_abs_min
        self.SROmax_abs_ppm = SROmax_abs_ppm
        self.Flag_DisplayResults = Flag_DisplayResults
        
        # Help parameters
        self.LowFreq_InpSig_fl_Hz = .01 * self.RefSampRate_fs_Hz / 2
        self.UppFreq_InpSig_fu_Hz = .95 * self.RefSampRate_fs_Hz / 2
        self.RateDXCPPhaT_Hz = self.RefSampRate_fs_Hz / self.FFTshift
        self.AccumTime_B_NumFr = \
            int(self.AccumTime_B_sec // (1 / self.RateDXCPPhaT_Hz))
        self.B_smpls = self.AccumTime_B_NumFr * self.FFTshift
        self.Upsilon = int(self.FFTsize / 2 - 1)
        self.Lambda = int(((self.B_smpls * self.SROmax_abs_ppm) // 1e6) + 1)
        self.Cont_NumFr = self.AccumTime_B_NumFr + 1
        self.InvShiftFac_NumFr = \
            int(self.FFTsize / self.FFTshift)
        self.ResetPeriod_NumFr = \
            int(self.ResetPeriod_sec // (1 / self.RateDXCPPhaT_Hz))
        self.FFT_Nyq = int(self.FFTsize / 2 + 1)
        self.FreqResol = self.RefSampRate_fs_Hz / self.FFTsize
        self.LowFreq_InpSig_fl_bin = \
            int(self.LowFreq_InpSig_fl_Hz // self.FreqResol)
        self.UppFreq_InpSig_fu_bin = \
            int(self.UppFreq_InpSig_fu_Hz // self.FreqResol)
        self.NyqDist_fu_bin = self.FFT_Nyq - self.UppFreq_InpSig_fu_bin    
        self.redFFTsize1 = self.FFTsize - self.FrameSize
        self.redFFTsize2 = self.FFTsize - self.LowFreq_InpSig_fl_bin + 1
        
        # Initialize state variables of DXCP-PhaT.
        # SRO estimate (updated once per signal segment)
        self.SROppm_est_seg = 0
        # STO estimate (updated once per signal segment)
        self.STOsmp_est_seg = 0
        # current SRO estimate
        self.SROppm_est_ell = 0
        # current STO estimate
        self.STOsmp_est_ell = 0
        # time offset between channels at the end of the current frame
        self.TimeOffsetEndSeg_est_out = 0
        # Averaged CSD with Phase Transform
        self.GCSD_PhaT_avg = np.zeros((self.FFTsize,1), dtype=complex)
        # Container with past GCSD_PhaT_avg values
        self.GCSD_PhaT_avg_Cont = \
            np.zeros((self.FFTsize, self.Cont_NumFr), dtype=complex)
        # averaged CSD-2
        self.GCSD2_avg = np.zeros((self.FFTsize,1), dtype=complex)
        # smoothed shifted first CCF
        self.GCCF1_smShftAvg = \
            np.zeros((2 * self.Upsilon + 1, 1), dtype=float)
        # input buffer of DXCP-PhaT-CL
        self.InputBuffer = np.zeros((self.FFTsize, 2), dtype=float)
        # counter for filling of the input buffer before executing DXCP-PhaT
        self.ell_execDXCPPhaT = 1
        # counter within signal sections with reset of CL estimator
        self.ell_inSigSec = 1
        # counter of signal sections
        self.ell_sigSec = np.int16(1)
        # counter of recursive simple averaging over CCF-2
        self.ell_GCCF2avg = 1
        # counter of recursive simple averaging over shifted CCF-1 (STO)
        self.ell_shftCCF1avg = 1
        
        # print an info after creation of DXCPPhaT instance in a help terminal
        print('... instance object for DXCP-PhaT created ...')
        sys.stdout.flush()
        
        # print --SmoConst_CSDPhaT_alpha parameter of DXCP-PhaT method
        #print('... DXCP-PhaT method: --SmoConst_CSDPhaT_alpha = %4.3f'% \
        #    (self.SmoConst_CSDPhaT_alpha))
        #sys.stdout.flush()


    """
    
    Process data of the current signal frame
    
    """
    def process_data(self, z_12_ell):

        # fill the internal buffer of DXCP-PhaT-CL (from right to left)
        self.InputBuffer[np.arange(self.redFFTsize1), :] = \
            self.InputBuffer[np.arange(self.FrameSize, self.FFTsize), :]
        self.InputBuffer[np.arange(self.redFFTsize1, self.FFTsize), :] = \
            z_12_ell

        # condition for execution of DXCP-PhaT
        if self.ell_execDXCPPhaT == int(self.FFTshift / self.FrameSize):
            # reset counter for filling of the input buffer
            self.ell_execDXCPPhaT = 0

            # state update of DXCP-PhaT-CL estimator
            self.dxcpphat_stateupdate()

        # update counter for filling of the input buffer
        self.ell_execDXCPPhaT += 1

        # calculate output of DXCP-PhaT estimator
        OutputDXCPPhaTcl = {}
        OutputDXCPPhaTcl['SROppm_est_out'] = self.SROppm_est_seg
        OutputDXCPPhaTcl['TimeOffsetEndSeg_est_out'] = \
            self.TimeOffsetEndSeg_est_out

        return OutputDXCPPhaTcl

    
    """
    
    Update state variables of DXCP-PhaT
    
    """
    def dxcpphat_stateupdate(self):
        # 0) reset counter ell_inSigSec for every new signal section
        if self.ell_inSigSec == self.ResetPeriod_NumFr + 1:
            self.ell_sigSec += 1
            self.ell_inSigSec = 1
            self.ell_GCCF2avg = 1
            self.ell_shftCCF1avg = 1

        # 1) Windowing to the current frames acc. to eq. (5) in [1]
        analWin = signal.blackman(self.FFTsize, sym=False)
        z_12_win = \
            self.InputBuffer * np.vstack((analWin, analWin)).transpose()

        # 2) Calculate generalized (normaized) GCSD with Phase Transform
        Z_12 = np.fft.fft(z_12_win, self.FFTsize, 0)
        Z_12_act = Z_12[:, 0] * np.conj(Z_12[:, 1])
        Z_12_act_abs = abs(Z_12_act)
        # avoid division by 0
        Z_12_act_abs[Z_12_act_abs < self.Z_12_abs_min] = self.Z_12_abs_min
        GCSD_PhaT_act = Z_12_act / Z_12_act_abs
        if self.ell_inSigSec == 1:  # if new signal section begins
            self.GCSD_PhaT_avg = GCSD_PhaT_act
        else:
            self.GCSD_PhaT_avg = \
                self.SmoConst_CSDPhaT_alpha * self.GCSD_PhaT_avg + \
                    (1 - self.SmoConst_CSDPhaT_alpha) * GCSD_PhaT_act

        # 3) Fill DXCP-container with Cont_NumFr number of past GCSD_PhaT_avg
        self.GCSD_PhaT_avg_Cont[:, np.arange(self.Cont_NumFr - 1)] = \
            self.GCSD_PhaT_avg_Cont[:, 1:]
        self.GCSD_PhaT_avg_Cont[:, (self.Cont_NumFr - 1)] = self.GCSD_PhaT_avg

        # 4) As soon as DXCP-container is filled with resampled data,
        # calculate the second GCSD based on last and first vectors
        # of DXCP-container and perform time averaging
        if self.ell_inSigSec >= self.Cont_NumFr + \
            (self.InvShiftFac_NumFr - 1) + self.AddContWait_NumFr:
            # Calculate the second GCSD
            GCSD2_act = self.GCSD_PhaT_avg_Cont[:, -1] * \
                np.conj(self.GCSD_PhaT_avg_Cont[:, 0])
            # simple averaging over the whole signal segment
            self.GCSD2_avg[:, 0] = \
                ((self.ell_GCCF2avg - 1) * self.GCSD2_avg[:, 0] + \
                    GCSD2_act) / self.ell_GCCF2avg
            self.ell_GCCF2avg += 1
            # remove components w.o. coherent components
            GCSD2_avg_ifft = self.GCSD2_avg
            # set lower frequency bins (w.o. coherent components) to 0
            GCSD2_avg_ifft[np.arange(self.LowFreq_InpSig_fl_bin), 0] = 0
            GCSD2_avg_ifft[np.arange(self.redFFTsize2, self.FFTsize), 0] = 0
            # set upper frequency bins (w.o. coherent components) to 0
            GCSD2_avg_ifft[np.arange(self.FFT_Nyq - self.NyqDist_fu_bin - 1, \
                self.FFT_Nyq + self.NyqDist_fu_bin), 0] = 0
            # Calculate averaged CCF-2 in time domain
            GCCF2_act1 = np.fft.ifft(GCSD2_avg_ifft, n=self.FFTsize, axis=0)
            GCCF2_avg_ell_big = np.fft.fftshift(np.real(GCCF2_act1))
            idx = np.arange(self.FFT_Nyq - self.Lambda - 1, \
                self.FFT_Nyq + self.Lambda)
            GCCF2avg_ell = GCCF2_avg_ell_big[idx, 0]

        # 5) Parabolic interpolation (13) with (14) with maximum search
        # as in [1] and calculation of the remaining current SRO estimate
        # sim. to (15) in [1]. As soon as GCSD2_avg is smoothed enough
        # in every section.
        if self.ell_inSigSec >= self.Cont_NumFr + (self.InvShiftFac_NumFr-1) \
            + self.AddContWait_NumFr + self.SettlingCSD2avg_NumFr:
            idx_max = GCCF2avg_ell.argmax(0)
            if (idx_max == 0) or (idx_max == 2 * self.Lambda):
                DelATSest_ell_frac = 0
            else:
                # set supporting points for y(x) for x={-1,0,1} for search
                # of real-valued maximum
                sup_pnts = GCCF2avg_ell[np.arange(idx_max - 1, idx_max + 2)]
                # calculate fractional of the maximum via x_max=-b/2/a
                # for y(x) = a*x^2 + b*x + c
                DelATSest_ell_frac = (sup_pnts[2, ] - sup_pnts[0, ]) \
                    / 2 / ( 2 * sup_pnts[1, ] - sup_pnts[2, ] - sup_pnts[0, ])
            # resulting real-valued x_max
            DelATSest_ell = idx_max - self.Lambda + DelATSest_ell_frac 
            self.SROppm_est_ell = DelATSest_ell / self.B_smpls * 10 ** 6

        # 6) STO-estimation after removing of SRO-induced time offset in CCF-1
        if self.ell_inSigSec >= self.Cont_NumFr + (self.InvShiftFac_NumFr-1) \
            + self.AddContWait_NumFr + self.SettlingCSD2avg_NumFr:
            # a) phase shifting of GCSD-1 to remove SRO-induced time offset
            timeOffset_forShift = self.SROppm_est_ell * 10 ** (-6) \
                * self.FFTshift * (self.ell_inSigSec - 1)
            idx = np.arange(self.FFTsize).transpose()
            expTerm = np.power(math.e, 1j * 2 * math.pi / self.FFTsize \
                * timeOffset_forShift * idx)
            GCSD1_smShft = self.GCSD_PhaT_avg * expTerm
            # b) remove components w.o. coherent components
            GCSD1_smShft_ifft = GCSD1_smShft
            # set lower frequency bins (w.o. coherent components) to 0
            GCSD1_smShft_ifft[np.arange(self.LowFreq_InpSig_fl_bin),] = 0
            GCSD1_smShft_ifft[np.arange(self.FFTsize - \
                self.LowFreq_InpSig_fl_bin + 1, self.FFTsize),] = 0
            # set upper frequency bins (w.o. coherent components) to 0
            GCSD1_smShft_ifft[np.arange(self.FFT_Nyq-self.NyqDist_fu_bin-1, \
                self.FFT_Nyq + self.NyqDist_fu_bin),] = 0
            # c) go into the time domain via calculation of shifted GCC-1
            GCSD1_act1 = np.fft.ifft(GCSD1_smShft_ifft, n=self.FFTsize)
            GCCF1_sroComp_big = np.fft.fftshift(np.real(GCSD1_act1))
            GCCF1_sroComp = GCCF1_sroComp_big[np.arange(self.FFT_Nyq - \
                self.Upsilon - 1, self.FFT_Nyq + self.Upsilon),]
            # d) averaging over time and zero-phase filtering within the frame
            self.GCCF1_smShftAvg[:, 0] = \
                ((self.ell_shftCCF1avg - 1) * self.GCCF1_smShftAvg[:, 0] \
                    + GCCF1_sroComp) / self.ell_shftCCF1avg
            GCCF1_smShftAvgAbs = np.abs(self.GCCF1_smShftAvg)
            self.ell_shftCCF1avg += 1
            # e) Maximum search over averaged filtered shifted GCC-1
            idx_max = GCCF1_smShftAvgAbs.argmax(0)
            if (idx_max == 0) or (idx_max == 2 * self.Upsilon):
                SSOsmp_est_ell_frac = 0
            else:
                # set supporting points for y(x) for x={-1,0,1}
                # for search of real-valued maximum
                sup_pnts = GCCF1_smShftAvgAbs[np.arange(idx_max-1, idx_max+2)]
                # calculate fractional of the maximum via x_max=-b/2/a
                # for y(x) = a*x^2 + b*x + c
                SSOsmp_est_ell_frac = (sup_pnts[2, ] - sup_pnts[0, ]) \
                        / 2 / (2 * sup_pnts[1,] - sup_pnts[2,] - sup_pnts[0,])
            # resulting real-valued x_max
            self.STOsmp_est_ell = idx_max - self.Upsilon + SSOsmp_est_ell_frac

        # 7) Update of SRO and STO estimates only at the end of signal segment
        if self.ell_inSigSec == self.ResetPeriod_NumFr:
            # resulting SRO estimate in the first signal section
            self.SROppm_est_seg = self.SROppm_est_ell
            # resulting STO estimate in the first signal section
            self.STOsmp_est_seg = self.STOsmp_est_ell
            # calculate a time offset between channels at the end
            # of the current signal segment
            self.TimeOffsetEndSeg_est_out = \
                self.STOsmp_est_ell + self.SROppm_est_ell * 10 ** (-6) \
                    * self.FFTshift * self.ResetPeriod_NumFr
            # for new signal segment set SRO and STO estimates to 0
            self.SROppm_est_ell = 0
            self.STOsmp_est_ell = 0
            # display estimates at the end of the current signal segment
            if self.Flag_DisplayResults == 1:
                print('%d. Sig-Seg: SRO_ppm=%6.3f; TimeOffset_smp=%6.3f'\
                    % (self.ell_sigSec, self.SROppm_est_seg, \
                        self.TimeOffsetEndSeg_est_out))
                sys.stdout.flush()

        # update counter of DXCP frames within a signal section
        self.ell_inSigSec += 1

