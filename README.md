# WASN
A collection of Demos for DFG FOR 2457 Acoustic Sensor Networks project

## Marvelo Utils
The [marvelo_utils](marvelo_utils/) package implements typical functions needed to implement a
demonstrator using the [MARVELO](https://github.com/CN-UPB/MARVELO) framework:
    
- [Audio Recording](marvelo_utils/marvelo_utils/audio_recording/): 
   - Read audio from a sound card
   - Write recorded audio to a file
- [Monitoring](marvelo_utils/marvelo_utils/monitoring/):
   - Send all results to another device on the network where they can be
   displayed
   - Live visualization of the results
- [Pipe Handling](marvelo_utils/marvelo_utils/pipe/):
   - Write data to a pipe connecting different modules
   - Read data from a pipe connecting different modules

## Overview
Acoustic sensor network (ASN) technologies enable the cooperation of smartphones, tablet computers, smart speakers, and other devices to form a virtual microphone array and a distributed processing system. Thus, ASNs do not only improve the acquisition of acoustic signals but also provide a flexible platform for novel applications. The successful deployment, however, requires a variety of technological solutions such as distributed signal enhancement and beamforming,
sample-accurate signal synchronization, methods for distributed machine learning and signal classification, and privacy-aware distributed processing techniques. We propose to present several RPi-based real-time demonstrations to showcase the progress achieved in the DFG-funded collaborative research project Acoustic Sensor Networks.

## Demos

  - Distributed Processing Plattform (H. Afifi, H. Karl) [[codes]](https://github.com/CN-UPB/MARVELO) [[video]](https://drive.google.com/file/d/1FhXeoun40hiU3WRceljSOrwO5wBEkZ5Y/view?usp=sharing): Current implementations of wireless acoustic sensor networks (e.g., Alexa and Google Assistant) rely on centralized architectures for signal processing. Previously, we presented the MARVELO framework for exploiting the processing capabilities of wireless smart devices (e.g., Raspberry Pi’s) for local signal processing. MARVELO adds an extra layer of privacy and avoids the round-trip delay over the internet. In this demo, we will show how to use MARVELO to collect information about the network and test the implementationon on local devices (e.g., a laptop), before deploying distributed signal processing on smart devices.
  - Signal Synchronization (A. Chinaev, G. Enzner): The typical ad-hoc nature of wireless acoustic sensor networks imposes the need to synchronize the continuous audio data streams of the sensor nodes. This requires an accurate estimation of the sampling rate offset (SRO) and the sampling time offset (STO), solely using the microphone signals. While the STO is induced by different starting times of recording threads, the SRO is caused by independent clocks of ad hoc A/D converters. We will demonstrate a novel Double-Cross-Correlation Processor with Phase Transform (DXCP-PhaT) [[paper]](https://ieeexplore.ieee.org/document/9399796) which estimates the offset parameters in a blind way only from real-time microphone recordings on a network of Raspberry Pis [[codes]](https://github.com/CN-UPB/WASN/tree/main/DXCPPhaT_demo) [[video]](https://drive.google.com/file/d/1PPQpSQuOYvzzx8aNCgrI8y_suy21Q7l_/view).
  - Acoustic Source Extraction (A. Brendel, W. Kellermann): We will demonstrate informed acoustic source extraction based on independent vector analysis (IVA). The extraction of a desiredsource from a mixture of acoustic sources in real time requires computationally efficient optimizationschemes for IVA as well as a reliable selection of the target source. Our recently published algorithm exploits knowledge on the direction of arrival of the target source to extract it from an observed mixture of multiple simultaneously active acoustic sources. The use of a background model for thedemixing system will allow for a computationally efficient implementation
  - Privacy-preserving Features (A. Nelus, R. Martin):  We will demonstrate the utility ofprivacy-preserving features in distributed audio feature extraction and classification tasks. We willcompare the privacy-preserving features to plain audio features in a gender recognition vs. speakeridentification task. We show that speaker identification can be suppressed while at the same timethe feature stream is still suitable for gender recognition.
  - Signal Synchronization and Sound Event Detection
    (J. Ebbers, T. Gburrek, J. Schmalenstroeer, R. Häb-Umbach):
    - Signal Synchronization: We will demonstrate the synchronization, i.e.
    sampling time offset (STO) and sampling rate offset (SRO) compensation,
    of two continuous audio data streams in real time. Hereby, the focus
    especially lies on an online SRO estimation and compensation. The SRO is
    estimated using our recently proposed online weighted average coherence
    drift (WACD) method. Furthermore, we demonstrate how the SRO estimates can
    be used to compensate for the SRO based on an online resampling using an
    STFT-resampling method. Finally, we show how signal synchronization and SED
    can profit from each other.
    - Sound Event Detection: We demonstrate an online sound event detection (SED).
    In the demo we will detect the sounds clapping, music, speech, telephone
    ringing and whistling. We use our forward-backward convolutional
    recurrent neural network (FBCRNN) model here, which is trained on the
    weakly labeled Audio Set data set [5].  At inference time it provides sound
    event scores once per second. We will here showcase a scenario with
    processing on several sensor nodes followed by a decision fusion to
    consolidate the detection results. For the decision fusion synchronous
    detection scores are required from the different sensor nodes, where we rely
    on the aforementioned signal synchronization.

## Acknoweledgment:


[Deutsche Forschungsgemeinschaft - DFG-FOR 2457](https://www.uni-paderborn.de/asn/)

![img](https://www.uni-paderborn.de/fileadmin/_processed_/9/2/csm_ASNLogo_c443ce161b.png)
