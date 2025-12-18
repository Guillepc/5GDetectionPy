#!/usr/bin/env python3
"""
Timing estimation module for 5G NR signals.
"""

import numpy as np

from py3gpp.nrPSS import nrPSS
from py3gpp.nrPSSIndices import nrPSSIndices
from py3gpp.nrTimingEstimate import nrTimingEstimate

# Global cache for PSS sequences and indices
_PSS_CACHE = {}
_PSS_INDICES_CACHE = None


def _get_cached_pss(nid2: int) -> np.ndarray:
    """Get or compute PSS sequence for a given NID2."""
    if nid2 not in _PSS_CACHE:
        _PSS_CACHE[nid2] = nrPSS(nid2)
    return _PSS_CACHE[nid2]


def _get_cached_pss_indices() -> np.ndarray:
    """Get or compute PSS indices (constant)."""
    global _PSS_INDICES_CACHE
    if _PSS_INDICES_CACHE is None:
        _PSS_INDICES_CACHE = nrPSSIndices()
    return _PSS_INDICES_CACHE


def estimate_timing_offset(waveform: np.ndarray, nid2: int, scs: int, 
                           sample_rate: float, verbose: bool = False) -> int:
    """
    Timing offset estimation using nrTimingEstimate.
    
    Args:
        waveform: IQ signal with frequency correction applied
        nid2: Detected PSS ID (0, 1 or 2)
        scs: Subcarrier spacing in kHz
        sample_rate: Sample rate in Hz
        verbose: If True, displays processing information
    
    Returns:
        timing_offset: Offset in samples from the start of the slot
    """
    if verbose:
        print("Timing offset estimation...")
    
    nrb_ssb = 20
    pss_indices = _get_cached_pss_indices()
    pss_seq = _get_cached_pss(nid2)
    
    # Create refGrid with PSS in symbol 2 (0-indexed: symbol 1)
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
    
    if verbose:
        print(f"  Timing offset: {timing_offset} samples")
    
    return timing_offset
