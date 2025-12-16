#!/usr/bin/env python3
"""
Demodulador 5G NR - Funciones principales de demodulación.
"""

import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any
import warnings

from scipy.io import loadmat
from py3gpp.nrOFDMDemodulate import nrOFDMDemodulate

try:
    import h5py
    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False
    warnings.warn("h5py no disponible. Solo se pueden leer .mat v7 y anteriores.")

from frequency_correction import frequency_correction_ofdm
from timing_estimation import estimate_timing_offset
from cell_detection import detect_cell_id_sss, detect_strongest_ssb
from visualization import plot_resource_grid, save_demodulation_log, save_error_log


def load_mat_file(filename: str) -> np.ndarray:
    """
    Carga waveform desde archivo .mat (v7 o v7.3 HDF5).
    
    Args:
        filename: Ruta al archivo .mat
    
    Returns:
        waveform: Array complejo con la señal IQ
    """
    try:
        mat_data = loadmat(filename)
        waveform = mat_data['waveform'].flatten()
        return waveform
    except Exception as e1:
        if not HAS_H5PY:
            raise RuntimeError(f"Archivo requiere h5py: {e1}")
        
        try:
            with h5py.File(filename, 'r') as f:
                wf_h5 = f['waveform'][()]
                if wf_h5.dtype.names:
                    waveform = wf_h5['real'] + 1j * wf_h5['imag']
                else:
                    waveform = wf_h5.view(complex)
                waveform = waveform.flatten()
                return waveform
        except Exception as e2:
            raise RuntimeError(f"Error leyendo .mat: v7 falló ({e1}), v7.3 falló ({e2})")


def demodulate_ssb(waveform: np.ndarray, 
                   scs: int = 30,
                   sample_rate: float = 19.5e6,
                   lmax: int = 8) -> Dict[str, Any]:
    """
    Demodula una señal SSB y detecta Cell ID.
    Función principal para uso desde otros scripts.
    
    Args:
        waveform: Señal IQ capturada
        scs: Subcarrier spacing en kHz (15 o 30)
        sample_rate: Sample rate en Hz
        lmax: Número de SSB bursts a procesar (típicamente 8)
    
    Returns:
        dict con:
            - cell_id: Physical Cell ID detectado
            - nid1, nid2: Componentes del Cell ID
            - strongest_ssb: Índice del SSB más fuerte
            - power_db: Potencia en dB
            - snr_db: SNR estimado en dB
            - freq_offset: Offset de frecuencia en Hz
            - timing_offset: Offset de timing en muestras
            - grid_display: Resource grid para visualización (540×54)
            - waveform_corrected: Waveform con correcciones aplicadas
    """
    # 1. Corrección de frecuencia
    search_bw = 3 * scs
    waveform_corrected, freq_offset, nid2 = frequency_correction_ofdm(
        waveform, scs, sample_rate, search_bw
    )
    
    # 2. Estimación de timing
    timing_offset = estimate_timing_offset(waveform_corrected, nid2, scs, sample_rate)
    waveform_aligned = waveform_corrected[timing_offset:]
    
    # 3. Demodulación OFDM del primer SSB
    nrb_ssb = 20
    n_symbols_ssb = 4
    nfft_ssb = 256
    
    mu = (scs // 15) - 1
    cp_lengths = np.zeros(14, dtype=int)
    for i in range(14):
        if i == 0 or i == 7 * 2**mu:
            cp_lengths[i] = int((144 * 2**(-mu) + 16) * (sample_rate / 30.72e6))
        else:
            cp_lengths[i] = int((144 * 2**(-mu)) * (sample_rate / 30.72e6))
    
    samples_per_ssb = sum([nfft_ssb + cp_lengths[i] for i in range(n_symbols_ssb)])
    waveform_ssb = waveform_aligned[:samples_per_ssb]
    
    grid_ssb = nrOFDMDemodulate(
        waveform=waveform_ssb,
        nrb=nrb_ssb,
        scs=scs,
        initialNSlot=0,
        CyclicPrefix='normal',
        Nfft=nfft_ssb,
        SampleRate=sample_rate
    )
    
    # 4. Detección de Cell ID
    nid1, max_corr = detect_cell_id_sss(grid_ssb, nid2)
    cell_id = 3 * nid1 + nid2
    
    # 5. Demodular todos los SSB bursts
    ssb_grids = np.zeros((nrb_ssb * 12, n_symbols_ssb, lmax), dtype=complex)
    samples_per_ssb_period = int(sample_rate * 0.02 / lmax)
    
    for i_ssb in range(lmax):
        start_idx = i_ssb * samples_per_ssb_period
        if start_idx + samples_per_ssb_period <= len(waveform_corrected):
            wf_ssb = waveform_corrected[start_idx:start_idx + samples_per_ssb_period]
            grid = nrOFDMDemodulate(
                waveform=wf_ssb,
                nrb=nrb_ssb,
                scs=scs,
                initialNSlot=0,
                CyclicPrefix='normal',
                Nfft=nfft_ssb,
                SampleRate=sample_rate
            )
            ssb_grids[:, :, i_ssb] = grid[:, :n_symbols_ssb]
    
    # 6. Detectar SSB más fuerte
    strongest_ssb, power_db, snr_db = detect_strongest_ssb(ssb_grids, nid2, nid1, lmax)
    
    # 7. Crear resource grid para visualización (45 RB)
    demod_rb = 45
    grid_full = nrOFDMDemodulate(
        waveform=waveform_aligned,
        nrb=demod_rb,
        scs=scs,
        initialNSlot=0,
        SampleRate=sample_rate
    )
    
    max_symbols = min(54, grid_full.shape[1])
    grid_display = grid_full[:, :max_symbols]
    
    return {
        'cell_id': cell_id,
        'nid1': nid1,
        'nid2': nid2,
        'strongest_ssb': strongest_ssb,
        'power_db': power_db,
        'snr_db': snr_db,
        'freq_offset': freq_offset,
        'timing_offset': timing_offset,
        'sss_correlation': max_corr,
        'grid_display': grid_display,
        'waveform_corrected': waveform_corrected
    }


def demodulate_file(mat_file: str, 
                   scs: int = 30,
                   gscn: int = 7929,
                   lmax: int = 8,
                   output_folder: Optional[str] = None,
                   save_plot: bool = True) -> Optional[Dict[str, Any]]:
    """
    Demodula un archivo .mat y opcionalmente guarda resultados.
    
    Args:
        mat_file: Ruta al archivo .mat
        scs: Subcarrier spacing en kHz
        gscn: GSCN del canal
        lmax: Número de SSB bursts
        output_folder: Carpeta para guardar resultados (None = no guardar)
        save_plot: Guardar imagen del resource grid
    
    Returns:
        Diccionario con resultados o None si falla
    """
    print("="*70)
    print(f"Demodulando: {Path(mat_file).name}")
    print("="*70)
    
    try:
        # Cargar waveform
        waveform = load_mat_file(mat_file)
        print(f"✓ Waveform cargado: {len(waveform)} muestras")
        
        # Demodular
        results = demodulate_ssb(waveform, scs=scs, lmax=lmax)
        
        # Añadir metadatos
        results['scs'] = scs
        results['sample_rate'] = 19.5e6
        results['gscn'] = gscn
        
        # Imprimir resultados
        print("\n" + "="*70)
        print("RESULTADOS")
        print("="*70)
        print(f"Cell ID: {results['cell_id']}")
        print(f"  NID1: {results['nid1']}")
        print(f"  NID2: {results['nid2']}")
        print(f"Strongest SSB: {results['strongest_ssb']}")
        print(f"Potencia: {results['power_db']:.1f} dB")
        print(f"SNR: {results['snr_db']:.1f} dB")
        print(f"Freq offset: {results['freq_offset']/1e3:.3f} kHz")
        print(f"Timing offset: {results['timing_offset']} muestras")
        print("="*70)
        
        # Guardar resultados si se especifica carpeta
        if output_folder is not None:
            file_name = Path(mat_file).stem
            
            if save_plot and 'grid_display' in results:
                plot_resource_grid(
                    results['grid_display'],
                    results['cell_id'],
                    results['snr_db'],
                    output_folder=output_folder,
                    filename=f'{file_name}_resource_grid'
                )
            
            save_demodulation_log(results, mat_file, output_folder, f'{file_name}_info')
        
        return results
        
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        
        if output_folder is not None:
            file_name = Path(mat_file).stem
            save_error_log(e, mat_file, output_folder, f'{file_name}_ERROR')
        
        return None


def demodulate_folder(folder_path: str,
                     scs: int = 30,
                     gscn: int = 7929,
                     lmax: int = 8,
                     output_folder: Optional[str] = None,
                     pattern: str = "*.mat") -> Dict[str, Any]:
    """
    Demodula todos los archivos .mat en una carpeta.
    
    Args:
        folder_path: Ruta a la carpeta con archivos .mat
        scs: Subcarrier spacing en kHz
        gscn: GSCN del canal
        lmax: Número de SSB bursts
        output_folder: Carpeta para guardar resultados
        pattern: Patrón de archivos a procesar
    
    Returns:
        Diccionario con estadísticas y resultados
    """
    folder = Path(folder_path)
    mat_files = sorted(folder.glob(pattern))
    
    print(f"Encontrados {len(mat_files)} archivos {pattern} en {folder_path}")
    
    results = []
    successful = 0
    failed = 0
    
    for mat_file in mat_files:
        result = demodulate_file(
            str(mat_file),
            scs=scs,
            gscn=gscn,
            lmax=lmax,
            output_folder=output_folder
        )
        
        if result is not None:
            results.append(result)
            successful += 1
        else:
            failed += 1
        
        print()  # Línea en blanco entre archivos
    
    print("="*70)
    print(f"Procesamiento completado: {successful} exitosos, {failed} fallidos")
    print("="*70)
    
    return {
        'results': results,
        'successful': successful,
        'failed': failed,
        'total': len(mat_files)
    }
