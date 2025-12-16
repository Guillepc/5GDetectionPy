#!/usr/bin/env python3
"""
Módulo de detección de Cell ID para señales 5G NR.
"""

import numpy as np
from typing import Tuple

from py3gpp.nrSSS import nrSSS
from py3gpp.nrSSSIndices import nrSSSIndices
from py3gpp.nrExtractResources import nrExtractResources
from py3gpp.nrPBCHDMRS import nrPBCHDMRS
from py3gpp.nrPBCHDMRSIndices import nrPBCHDMRSIndices


def detect_cell_id(ssb_grid: np.ndarray, nid2: int, verbose: bool = False) -> Tuple[int, float]:
    """
    Detecta el Cell ID usando SSS.
    
    Args:
        ssb_grid: Resource grid del SSB (240 subportadoras × 4 símbolos)
        nid2: PSS ID detectado (0, 1 o 2)
        verbose: Si True, muestra información del procesamiento
    
    Returns:
        nid1: Physical cell ID group (0-335)
        max_corr: Valor de correlación máxima
    """
    if verbose:
        print("Detección de Cell ID (SSS)...")
    
    sss_indices = nrSSSIndices().astype(int)
    sss_rx = nrExtractResources(sss_indices, ssb_grid)
    
    correlations = np.zeros(336)
    for nid1 in range(336):
        cell_id = 3 * nid1 + nid2
        sss_ref = nrSSS(cell_id)
        correlation = sss_rx * np.conj(sss_ref)
        correlations[nid1] = np.sum(np.abs(correlation)**2)
    
    best_nid1 = int(np.argmax(correlations))
    max_corr = correlations[best_nid1]
    
    if verbose:
        print(f"  NID1 detectado: {best_nid1}")
        print(f"  Cell ID: {3 * best_nid1 + nid2}")
        print(f"  Correlación: {max_corr:.2f}")
    
    return best_nid1, max_corr


def detect_strongest_ssb(ssb_grids: np.ndarray, nid2: int, nid1: int, 
                         lmax: int = 8, verbose: bool = False) -> Tuple[int, float, float]:
    """
    Detecta el SSB más fuerte entre los Lmax candidatos.
    
    Args:
        ssb_grids: Array de grids SSB (240 × 4 × Lmax)
        nid2: PSS ID
        nid1: Physical cell ID group
        lmax: Número de SSB bursts a evaluar
        verbose: Si True, muestra información del procesamiento
    
    Returns:
        strongest_ssb: Índice del SSB más fuerte (0-7)
        power_db: Potencia en dB
        snr_db: SNR estimado en dB
    """
    if verbose:
        print(f"Detección de SSB más fuerte (Lmax={lmax})...")
    
    cell_id = 3 * nid1 + nid2
    sss_indices = nrSSSIndices()
    pbch_dmrs_indices = nrPBCHDMRSIndices(cell_id)
    
    powers = np.zeros(lmax)
    snrs = np.zeros(lmax)
    
    for i_ssb in range(lmax):
        grid = ssb_grids[:, :, i_ssb]
        
        # Potencia del SSS
        sss_rx = nrExtractResources(sss_indices, grid)
        powers[i_ssb] = np.mean(np.abs(sss_rx)**2)
        
        # SNR usando PBCH-DMRS
        try:
            dmrs_rx = nrExtractResources(pbch_dmrs_indices, grid)
            dmrs_ref = nrPBCHDMRS(cell_id, i_ssb)
            
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
        print(f"  SSB más fuerte: {strongest_ssb}")
        print(f"  Potencia: {power_db:.1f} dB")
        print(f"  SNR: {snr_db:.1f} dB")
    
    return strongest_ssb, power_db, snr_db
