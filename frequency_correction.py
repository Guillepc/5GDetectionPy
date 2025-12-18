#!/usr/bin/env python3
"""
Frequency correction module for 5G NR signals.
"""

import numpy as np
from scipy.signal import resample, oaconvolve
from typing import Tuple

from py3gpp.nrPSS import nrPSS
from py3gpp.nrPSSIndices import nrPSSIndices
from py3gpp.nrOFDMModulate import nrOFDMModulate

# Global cache for PSS sequences, indices, and reference grids
_PSS_CACHE = {}
_PSS_INDICES_CACHE = None
_REF_GRID_CACHE = {}
_REF_WAVEFORM_CACHE = {}


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


def _get_cached_ref_waveform(nid2: int, scs: int, sync_sr: float, sync_nfft: int) -> np.ndarray:
    """Get or compute reference waveform for a given NID2 and parameters."""
    key = (nid2, scs, sync_sr, sync_nfft)
    if key not in _REF_WAVEFORM_CACHE:
        nrb_ssb = 20
        pss_indices = _get_cached_pss_indices()
        ref_grid = np.zeros((nrb_ssb * 12, 4), dtype=complex)
        ref_grid[pss_indices, 0] = _get_cached_pss(nid2)
        ref_waveform, _ = nrOFDMModulate(
            grid=ref_grid,
            scs=scs,
            initialNSlot=0,
            SampleRate=sync_sr,
            Nfft=sync_nfft
        )
        _REF_WAVEFORM_CACHE[key] = ref_waveform
    return _REF_WAVEFORM_CACHE[key]


def frequency_correction_ofdm(waveform: np.ndarray, scs: int, sample_rate: float, 
                               search_bw: float, verbose: bool = False) -> Tuple[np.ndarray, float, int]:
    """
    Frequency correction and PSS detection using OFDM modulation.
    
    Args:
        waveform: Captured IQ signal
        scs: Subcarrier spacing in kHz (typically 30)
        sample_rate: Sample rate in Hz (typically 19.5e6)
        search_bw: Search bandwidth in kHz (typically 3*scs = 90)
        verbose: Display detailed processing information
    
    Returns:
        waveform_corrected: Waveform with frequency correction
        freq_offset: Detected frequency offset in Hz
        nid2: Detected PSS ID (0, 1 or 2)
    """
    if verbose:
        print("Frequency correction and PSS detection (OFDM method)...")
    
    # Synchronization parameters
    sync_nfft = 256
    sync_sr = sync_nfft * scs * 1000  # 256 * 30 * 1000 = 7.68 MHz
    
    # Coarse and fine search
    coarse_fshifts = np.arange(-search_bw, search_bw + scs, scs) * 1e3 / 2
    fine_fshifts = np.arange(-scs, scs + 1, 1) * 1e3 / 2
    fshifts = np.unique(np.concatenate([coarse_fshifts, fine_fshifts]))
    fshifts = np.sort(fshifts)
    
    peak_values = np.zeros((len(fshifts), 3))
    t = np.arange(len(waveform)) / sample_rate
    
    if verbose:
        print(f"  Testing {len(fshifts)} frequency offsets × 3 NID2...")
    
    for f_idx, f_shift in enumerate(fshifts):
        waveform_corrected = waveform * np.exp(-1j * 2 * np.pi * f_shift * t)
        num_samples_ds = int(len(waveform_corrected) * sync_sr / sample_rate)
        waveform_ds = resample(waveform_corrected, num_samples_ds)
        
        for nid2 in range(3):
            try:
                ref_waveform = _get_cached_ref_waveform(nid2, scs, sync_sr, sync_nfft)
                
                max_samples = min(len(waveform_ds), 300000)
                corr = oaconvolve(waveform_ds[:max_samples], 
                                 np.conj(ref_waveform[::-1]), mode='valid')
                peak_values[f_idx, nid2] = np.max(np.abs(corr))
            except Exception as e:
                peak_values[f_idx, nid2] = 0
    
    # Normalize and display results
    peak_values_norm = peak_values / np.max(peak_values) if np.max(peak_values) > 0 else peak_values
    
    if verbose:
        print(f"\n  PSS correlation matrix (normalized):")
        print(f"  {'Freq (kHz)':>12} {'NID2=0':>12} {'NID2=1':>12} {'NID2=2':>12}")
        for i, f in enumerate(fshifts):
            print(f"  {f/1e3:>12.2f} {peak_values_norm[i, 0]:>12.3f} {peak_values_norm[i, 1]:>12.3f} {peak_values_norm[i, 2]:>12.3f}")
    
    best_f_idx, best_nid2 = np.unravel_index(np.argmax(peak_values), peak_values.shape)
    freq_offset = fshifts[best_f_idx]
    
    if verbose:
        print(f"\n  → Detected NID2: {best_nid2}")
        print(f"  → Frequency offset: {freq_offset/1e3:.3f} kHz")
        print(f"  → Maximum peak: {peak_values[best_f_idx, best_nid2]:.2f}")
    
    waveform_corrected = waveform * np.exp(-1j * 2 * np.pi * freq_offset * t)
    
    return waveform_corrected, freq_offset, best_nid2
