#!/usr/bin/env python3
"""
Cell ID detection module for 5G NR signals.
"""

import numpy as np
from typing import Tuple

from py3gpp.nrSSS import nrSSS
from py3gpp.nrSSSIndices import nrSSSIndices
from py3gpp.nrExtractResources import nrExtractResources
from py3gpp.nrPBCHDMRS import nrPBCHDMRS
from py3gpp.nrPBCHDMRSIndices import nrPBCHDMRSIndices

# Global cache for sequences
_SSS_CACHE = {}
_PBCH_DMRS_CACHE = {}
_SSS_INDICES_CACHE = None
_PBCH_DMRS_INDICES_CACHE = {}


def _get_cached_sss(cell_id: int) -> np.ndarray:
    """Get or compute SSS sequence for a given cell ID."""
    if cell_id not in _SSS_CACHE:
        _SSS_CACHE[cell_id] = nrSSS(cell_id)
    return _SSS_CACHE[cell_id]


def _get_cached_sss_indices() -> np.ndarray:
    """Get or compute SSS indices (constant)."""
    global _SSS_INDICES_CACHE
    if _SSS_INDICES_CACHE is None:
        _SSS_INDICES_CACHE = nrSSSIndices().astype(int)
    return _SSS_INDICES_CACHE


def _get_cached_pbch_dmrs(cell_id: int, i_ssb: int) -> np.ndarray:
    """Get or compute PBCH DMRS sequence."""
    key = (cell_id, i_ssb)
    if key not in _PBCH_DMRS_CACHE:
        _PBCH_DMRS_CACHE[key] = nrPBCHDMRS(cell_id, i_ssb)
    return _PBCH_DMRS_CACHE[key]


def _get_cached_pbch_dmrs_indices(cell_id: int) -> np.ndarray:
    """Get or compute PBCH DMRS indices."""
    if cell_id not in _PBCH_DMRS_INDICES_CACHE:
        _PBCH_DMRS_INDICES_CACHE[cell_id] = nrPBCHDMRSIndices(cell_id)
    return _PBCH_DMRS_INDICES_CACHE[cell_id]


def detect_cell_id(ssb_grid: np.ndarray, nid2: int, verbose: bool = False) -> Tuple[int, float]:
    """
    Detects the Cell ID using SSS.
    
    Args:
        ssb_grid: SSB resource grid (240 subcarriers × 4 symbols)
        nid2: Detected PSS ID (0, 1 or 2)
        verbose: If True, displays processing information
    
    Returns:
        nid1: Physical cell ID group (0-335)
        max_corr: Maximum correlation value
    """
    if verbose:
        print("Cell ID detection (SSS)...")
    
    sss_indices = _get_cached_sss_indices()
    sss_rx = nrExtractResources(sss_indices, ssb_grid)
    
    correlations = np.zeros(336)
    for nid1 in range(336):
        cell_id = 3 * nid1 + nid2
        sss_ref = _get_cached_sss(cell_id)
        correlation = sss_rx * np.conj(sss_ref)
        correlations[nid1] = np.sum(np.abs(correlation)**2)
    
    best_nid1 = int(np.argmax(correlations))
    max_corr = correlations[best_nid1]
    
    if verbose:
        print(f"  Detected NID1: {best_nid1}")
        print(f"  Cell ID: {3 * best_nid1 + nid2}")
        print(f"  Correlation: {max_corr:.2f}")
    
    return best_nid1, max_corr


def detect_strongest_ssb(ssb_grids: np.ndarray, nid2: int, nid1: int, 
                         lmax: int = 8, verbose: bool = False) -> Tuple[int, float, float]:
    """
    Detects the strongest SSB among Lmax candidates.
    
    Args:
        ssb_grids: Array of SSB grids (240 × 4 × Lmax)
        nid2: PSS ID
        nid1: Physical cell ID group
        lmax: Number of SSB bursts to evaluate
        verbose: If True, displays processing information
    
    Returns:
        strongest_ssb: Index of the strongest SSB (0-7)
        power_db: Power in dB
        snr_db: Estimated SNR in dB
    """
    if verbose:
        print(f"Strongest SSB detection (Lmax={lmax})...")
    
    cell_id = 3 * nid1 + nid2
    sss_indices = _get_cached_sss_indices()
    pbch_dmrs_indices = _get_cached_pbch_dmrs_indices(cell_id)
    
    powers = np.zeros(lmax)
    snrs = np.zeros(lmax)
    
    for i_ssb in range(lmax):
        grid = ssb_grids[:, :, i_ssb]
        
        # SSS power
        sss_rx = nrExtractResources(sss_indices, grid)
        powers[i_ssb] = np.mean(np.abs(sss_rx)**2)
        
        # SNR usando PBCH-DMRS
        try:
            dmrs_rx = nrExtractResources(pbch_dmrs_indices, grid)
            dmrs_ref = _get_cached_pbch_dmrs(cell_id, i_ssb)
            
            if len(dmrs_rx) > 0 and len(dmrs_ref) > 0:
                h_est = dmrs_rx / dmrs_ref
                signal_power = np.mean(np.abs(h_est)**2)
                noise_power = np.var(np.abs(h_est - np.mean(h_est))**2)
                snrs[i_ssb] = signal_power / max(noise_power, 1e-10)
            else:
                snrs[i_ssb] = 0
        except:
            snrs[i_ssb] = 0
    
    strongest_ssb = int(np.argmax(powers))
    power_db = 10 * np.log10(powers[strongest_ssb] + 1e-12)
    snr_db = 10 * np.log10(snrs[strongest_ssb] + 1e-12)
    
    if verbose:
        print(f"  Strongest SSB: {strongest_ssb}")
        print(f"  Power: {power_db:.1f} dB")
        print(f"  SNR: {snr_db:.1f} dB")
    
    return strongest_ssb, power_db, snr_db
