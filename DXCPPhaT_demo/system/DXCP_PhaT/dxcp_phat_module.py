"""

MARVELO module for the Double-Cross-Correlation Processor (DXCP) with phase
transform (PhaT) for a blind Sampling-Rate Offset (SRO) and Time-Offset (STO)
estimation from [1].

[1] Chinaev A., Thuene P., and Enzner G., 'Double-Cross-Correlation Processing
for Blind Sampling-Rate and Time-Offset Estimation', IEEE/ACM Trans. on Audio,
Speech, and Language Proc., vol. 29, pp. 1881-1896, 2021.

"""

#!/usr/bin/env python3.6
import sys
import argparse
import numpy as np
from dxcp_phat import DXCPPhaT
from marvelo_utils.pipe.reader import PipeReader
from marvelo_utils.pipe.writer import PipeWriter
from scipy.io.wavfile import write


def parse_arguments():
    parser = argparse.ArgumentParser(description='arguments')
    parser.add_argument("--record", "-r", default=1, type=int)
    parser.add_argument("--segs2save", "-s", default=3, type=int)
    parser.add_argument("--inputs", "-i",  action="append")
    parser.add_argument("--outputs", "-o", action="append")
    parser.add_argument("--logfiles", "-l", action="append")
    
    # DXCP-PhaT Parameters
    parser.add_argument('--RefSampRate_fs_Hz', default = \
        DXCPPhaT.defaultParams['RefSampRate_fs_Hz'], type=int)
    parser.add_argument('--FrameSize', default = \
        DXCPPhaT.defaultParams['FrameSize'], type=int)
    parser.add_argument('--FFTshift', default = \
        DXCPPhaT.defaultParams['FFTshift'], type=int)
    parser.add_argument('--FFTsize', default = \
        DXCPPhaT.defaultParams['FFTsize'], type=int)
    parser.add_argument('--AccumTime_B_sec', default = \
        DXCPPhaT.defaultParams['AccumTime_B_sec'], type=float)
    parser.add_argument('--ResetPeriod_sec', default = \
        DXCPPhaT.defaultParams['ResetPeriod_sec'], type=float)
    parser.add_argument('--SmoConst_CSDPhaT_alpha', default = \
        DXCPPhaT.defaultParams['SmoConst_CSDPhaT_alpha'], type=float)
    parser.add_argument('--AddContWait_NumFr', default = \
        DXCPPhaT.defaultParams['AddContWait_NumFr'], type=int)
    parser.add_argument('--SettlingCSD2avg_NumFr', default = \
        DXCPPhaT.defaultParams['SettlingCSD2avg_NumFr'], type=int)
    parser.add_argument('--Z_12_abs_min', default = \
        DXCPPhaT.defaultParams['Z_12_abs_min'], type=float)
    parser.add_argument('--SROmax_abs_ppm', default = \
        DXCPPhaT.defaultParams['SROmax_abs_ppm'], type=float)
    
    return parser.parse_args()

# Parse input arguments
args = parse_arguments()

# print --segs2save parameter of MARVELO module for DXCP-PhaT
#print('... DXCP-PhaT module: --segs2save = %d' % (args.segs2save))
#sys.stdout.flush()

# Create an instance of DXCPPhaT class
Inst_DXCPPhaT = DXCPPhaT(
    RefSampRate_fs_Hz = args.RefSampRate_fs_Hz, 
    FrameSize = args.FrameSize, 
    FFTshift = args.FFTshift, 
    FFTsize = args.FFTsize, 
    AccumTime_B_sec = args.AccumTime_B_sec, 
    ResetPeriod_sec = args.ResetPeriod_sec, 
    SmoConst_CSDPhaT_alpha = args.SmoConst_CSDPhaT_alpha, 
    AddContWait_NumFr = args.AddContWait_NumFr, 
    SettlingCSD2avg_NumFr = args.SettlingCSD2avg_NumFr, 
    Z_12_abs_min = args.Z_12_abs_min, 
    SROmax_abs_ppm = args.SROmax_abs_ppm
)

# Create input and output pipes
readers = [PipeReader(int(args.inputs[0]), int(Inst_DXCPPhaT.FrameSize)),
               PipeReader(int(args.inputs[1]), int(Inst_DXCPPhaT.FrameSize))]
for reader in readers:
    reader.start()
writer_sro = PipeWriter([args.outputs[0]])
writer_delay = PipeWriter([args.outputs[1]])

# Main loop
flagSave = 1
inputData = np.float32(np.zeros((Inst_DXCPPhaT.FrameSize, 2)))
while True:
    # get input data from input pipes
    inputData[:, 0] = readers[0].get_next_block()
    inputData[:, 1] = readers[1].get_next_block()

    # execute DXCP-PhaT for one data frame inputData = (z_1[ell]; z_2[ell])
    OutputDXCPPhaT = Inst_DXCPPhaT.process_data(np.float32(inputData))

    # set output vectors of this wrapper-function
    SROppm_est_out = np.float32(OutputDXCPPhaT['SROppm_est_out'])
    TimeOffsetEndSeg_est_out = \
        np.float32(OutputDXCPPhaT['TimeOffsetEndSeg_est_out'])

    # write output data into output pipe
    outputData = \
        np.array([SROppm_est_out, TimeOffsetEndSeg_est_out], dtype=np.float32)
    writer_sro.write_next_block(SROppm_est_out*1e-6)
    writer_delay.write_next_block(TimeOffsetEndSeg_est_out)

    if (int(args.record) == 1) and \
        (Inst_DXCPPhaT.ell_sigSec <= args.segs2save):
        if Inst_DXCPPhaT.ell_inSigSec == 1:  # only once in the beginning
            frames = inputData
        else:
            frames = np.concatenate((frames, inputData))
        # save stereo signal of the last segment
        if (Inst_DXCPPhaT.ell_sigSec==args.segs2save) and \
            (Inst_DXCPPhaT.ell_inSigSec==Inst_DXCPPhaT.ResetPeriod_NumFr+1) \
             and (flagSave==1):
            WavOutput_FolderName = '/home/asn/asn_daemon/logfiles/'
            WavOutput_FileName = 'AsyncAudio2x1ch_' \
                + str(Inst_DXCPPhaT.ell_sigSec) + 'sigSegs.wav'
            write(WavOutput_FolderName + WavOutput_FileName, \
                Inst_DXCPPhaT.RefSampRate_fs_Hz, frames)
            flagSave = 0
            print('... ' + WavOutput_FileName + ' is saved :-)')
            sys.stdout.flush()
