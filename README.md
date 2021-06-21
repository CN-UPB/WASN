# WASN
A collection of Demos for DFG FOR 2457 Acoustic Sensor Networks project

## Overview
Acoustic sensor network (ASN) technologies enable the cooperation of smartphones, tablet
computers, smart speakers, and other devices to form a virtual microphone array and a distributed
processing system. Thus, ASNs do not only improve the acquisition of acoustic signals but also
provide a flexible platform for novel applications. The successful deployment, however, requires
a variety of technological solutions such as distributed signal enhancement and beamforming,
sample-accurate signal synchronization, methods for distributed machine learning and signal
classification, and privacy-aware distributed processing techniques.
We propose to present several RPi-based real-time demonstrations to showcase the progress achieved
in the DFG-funded collaborative research project Acoustic Sensor Networks. In case of a virtual
conference we will prepare and present demonstration videos.


## Demos

  - Distributed Processing Plattform (@haithamafifi, H. Karl): Current implementations of wirelessacoustic sensor networks (e.g., Alexa and Google Assistant) rely on centralized architectures forsignal processing. Previously, we presented the MARVELO framework for exploiting the processingcapabilities of wireless smart devices (e.g., Raspberry Pi’s) for local signal processing. MARVELOadds an extra layer of privacy and avoids the round-trip delay over the internet. In this demo, we willshow how to use MARVELO to collect information about the network and test the implementationon local devices (e.g., a laptop), before deploying distributed signal processing on smart devices.
  - Signal Synchronization: The typical ad-hoc nature of wireless acoustic sensor networks im-poses the need to synchronize the continuous audio data streams of the sensor nodes. This requiresan accurate estimation of the sampling rate offset (SRO) and the sampling time offset (STO), solelyusing the microphone signals. While the STO is induced by different starting times of recordingthreads, the SRO is caused by independent clocks of ad hoc A/D converters.
    - Demo B.1 (A. Chinaev, G. Enzner): We will demonstrate a novel Double-Cross-Correlation Pro-cessor (DXCP) which estimates the offset parameters in a blind way only from real-timemicrophone recordings on a network of Raspberry Pis.
    - Demo B.2 (@gburrek, J. Schmalenströer): We will demonstrate a newly proposed online weightedaveraged coherence drift (WACD) method. To this end, the SRO estimates are used togetherwith an initial estimate of the STO to synchronize the data streams by resampling.
  - C)Acoustic Source Extraction(A. Brendel, W. Kellermann): We will demonstrate informed acoustic source extraction based on independent vector analysis (IVA). The extraction of a desiredsource from a mixture of acoustic sources in real time requires computationally efficient optimizationschemes for IVA as well as a reliable selection of the target source. Our recently published algorithmexploits knowledge on the direction of arrival of the target source to extract it from an observedmixture of multiple simultaneously active acoustic sources. The use of a background model for thedemixing system will allow for a computationally efficient implementation
  - Privacy-preserving Features(A. Nelus, R. Martin):  We will demonstrate the utility ofprivacy-preserving features in distributed audio feature extraction and classification tasks. We willcompare the privacy-preserving features to plain audio features in a gender recognition vs. speakeridentification task. We show that speaker identification can be suppressed while at the same timethe feature stream is still suitable for gender recognition.
  - Sound Event Detection(@JanekEbb, R. Häb-Umbach): We will demonstrate sound eventdetection (SED) on an acoustic sensor network.  The algorithm will classify at least the eventclasses music, clapping, whistling, and possibly more classes, using a multi-label classificationapproach which allows for the recognition of simultaneously active events.  The system will betrained employing weakly labeled learning techniques to leverage unlabeled data, and it will buildon data augmentation to further improve recognition performance. We will showcase a scenariowith processing on several sensor nodes and decision fusion to consolidate the classification results.

## Acknoweledgment:


[Deutsche Forschungsgemeinschaft - DFG-FOR 2457](https://www.uni-paderborn.de/asn/)

![img](https://www.uni-paderborn.de/fileadmin/_processed_/9/2/csm_ASNLogo_c443ce161b.png)
