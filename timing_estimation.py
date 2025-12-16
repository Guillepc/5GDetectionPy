#!/usr/bin/env python3
"""
Módulo de estimación de timing para señales 5G NR.
"""

import numpy as np

from py3gpp.nrPSS import nrPSS
from py3gpp.nrPSSIndices import nrPSSIndices
from py3gpp.nrTimingEstimate import nrTimingEstimate


def estimate_timing_offset(waveform: np.ndarray, nid2: int, scs: int, 
                           sample_rate: float) -> int:
    """
    Estimación de timing offset usando nrTimingEstimate.
    
    Args:
        waveform: Señal IQ con corrección de frecuencia aplicada
        nid2: PSS ID detectado (0, 1 o 2)
        scs: Subcarrier spacing en kHz
        sample_rate: Sample rate en Hz
    
    Returns:
        timing_offset: Offset en muestras desde el inicio del slot
    """
    print("Estimación de timing offset...")
    
    nrb_ssb = 20
    pss_indices = nrPSSIndices()
    pss_seq = nrPSS(nid2)
    
    # Crear refGrid con PSS en el símbolo 2 (0-indexed: símbolo 1)
    ref_grid = np.zeros((nrb_ssb * 12, 2), dtype=complex)
    ref_grid[pss_indices.astype(int), 1] = pss_seq
    
    timing_offset = nrTimingEstimate(
        waveform=waveform,
        nrb=nrb_ssb,
        scs=scs,
        initialNSlot=0,
        refGrid=ref_grid,
        SampleRate=sample_rate
    )
    
    print(f"  Timing offset: {timing_offset} muestras")
    
    return timing_offset
