#!/usr/bin/env python3
"""
Módulo de corrección de frecuencia para señales 5G NR.
"""

import numpy as np
from scipy import signal as scipy_signal
from typing import Tuple

from py3gpp.nrPSS import nrPSS
from py3gpp.nrPSSIndices import nrPSSIndices
from py3gpp.nrOFDMModulate import nrOFDMModulate


def frequency_correction_ofdm(waveform: np.ndarray, scs: int, sample_rate: float, 
                               search_bw: float, verbose: bool = False) -> Tuple[np.ndarray, float, int]:
    """
    Corrección de frecuencia y detección de PSS usando OFDM modulation.
    
    Args:
        waveform: Señal IQ capturada
        scs: Subcarrier spacing en kHz (típicamente 30)
        sample_rate: Sample rate en Hz (típicamente 19.5e6)
        search_bw: Ancho de búsqueda en kHz (típicamente 3*scs = 90)
        verbose: Mostrar información detallada del procesamiento
    
    Returns:
        waveform_corrected: Waveform con corrección de frecuencia
        freq_offset: Offset de frecuencia detectado en Hz
        nid2: PSS ID detectado (0, 1 o 2)
    """
    if verbose:
        print("Corrección de frecuencia y detección PSS (método OFDM)...")
    
    # Parámetros de sincronización
    sync_nfft = 256
    sync_sr = sync_nfft * scs * 1000  # 256 * 30 * 1000 = 7.68 MHz
    nrb_ssb = 20  # SSB son 20 RBs = 240 subportadoras
    
    # PSS indices
    pss_indices = nrPSSIndices()
    
    # Crear grids de referencia para los 3 NID2
    ref_grids = np.zeros((nrb_ssb * 12, 4, 3), dtype=complex)
    for nid2 in range(3):
        ref_grids[pss_indices, 0, nid2] = nrPSS(nid2)
    
    # Búsqueda gruesa y fina
    coarse_fshifts = np.arange(-search_bw, search_bw + scs, scs) * 1e3 / 2
    fine_fshifts = np.arange(-scs, scs + 1, 1) * 1e3 / 2
    fshifts = np.unique(np.concatenate([coarse_fshifts, fine_fshifts]))
    fshifts = np.sort(fshifts)
    
    peak_values = np.zeros((len(fshifts), 3))
    t = np.arange(len(waveform)) / sample_rate
    
    if verbose:
        print(f"  Probando {len(fshifts)} offsets de frecuencia × 3 NID2...")
    
    for f_idx, f_shift in enumerate(fshifts):
        waveform_corrected = waveform * np.exp(-1j * 2 * np.pi * f_shift * t)
        num_samples_ds = int(len(waveform_corrected) * sync_sr / sample_rate)
        waveform_ds = scipy_signal.resample(waveform_corrected, num_samples_ds)
        
        for nid2 in range(3):
            try:
                ref_grid_nid2 = ref_grids[:, :, nid2]
                ref_waveform, _ = nrOFDMModulate(
                    grid=ref_grid_nid2,
                    scs=scs,
                    initialNSlot=0,
                    SampleRate=sync_sr,
                    Nfft=sync_nfft
                )
                
                max_samples = min(len(waveform_ds), 300000)
                corr = scipy_signal.correlate(waveform_ds[:max_samples], 
                                            ref_waveform, mode='valid')
                peak_values[f_idx, nid2] = np.max(np.abs(corr))
            except Exception as e:
                peak_values[f_idx, nid2] = 0
    
    # Normalizar y mostrar resultados
    peak_values_norm = peak_values / np.max(peak_values) if np.max(peak_values) > 0 else peak_values
    
    if verbose:
        print(f"\n  Matriz de correlaciones PSS (normalizadas):")
        print(f"  {'Freq (kHz)':>12} {'NID2=0':>12} {'NID2=1':>12} {'NID2=2':>12}")
        for i, f in enumerate(fshifts):
            print(f"  {f/1e3:>12.2f} {peak_values_norm[i, 0]:>12.3f} {peak_values_norm[i, 1]:>12.3f} {peak_values_norm[i, 2]:>12.3f}")
    
    best_f_idx, best_nid2 = np.unravel_index(np.argmax(peak_values), peak_values.shape)
    freq_offset = fshifts[best_f_idx]
    
    if verbose:
        print(f"\n  → NID2 detectado: {best_nid2}")
        print(f"  → Offset de frecuencia: {freq_offset/1e3:.3f} kHz")
        print(f"  → Pico máximo: {peak_values[best_f_idx, best_nid2]:.2f}")
    
    waveform_corrected = waveform * np.exp(-1j * 2 * np.pi * freq_offset * t)
    
    return waveform_corrected, freq_offset, best_nid2
