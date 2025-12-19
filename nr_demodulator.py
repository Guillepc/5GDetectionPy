#!/usr/bin/env python3
"""
5G NR Demodulator - Main demodulation functions.
"""

import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

from scipy.io import loadmat
from py3gpp.nrOFDMDemodulate import nrOFDMDemodulate
import time

try:
    import h5py
    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False
    warnings.warn("h5py not available. Only .mat v7 and earlier can be read.")

from frequency_correction import frequency_correction_ofdm
from frequency_correction import frequency_correction_ofdm_fast
from timing_estimation import estimate_timing_offset
from cell_detection import detect_cell_id, detect_strongest_ssb
from visualization import (plot_resource_grid, save_demodulation_log, save_error_log,
                           init_processing_log, append_success_to_log, 
                           append_error_to_log, finalize_processing_log)
from config_loader import get_config


def load_mat_file(filename: str) -> np.ndarray:
    """
    Loads waveform from .mat file (v7 or v7.3 HDF5).
    
    Args:
        filename: Path to .mat file
    
    Returns:
        waveform: Complex array with IQ signal
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
            raise RuntimeError(f"Error reading .mat: v7 failed ({e1}), v7.3 failed ({e2})")


def demodulate_ssb(waveform: np.ndarray,
                   scs: Optional[int] = None,
                   sample_rate: Optional[float] = None,
                   lmax: int = 8,
                   n_symbols_display: Optional[int] = None,
                   verbose: bool = False,
                   fast_mode: bool = False,
                   ssb_expected_position_ms: Optional[float] = None) -> Dict[str, Any]:
    """
    Demodulates an SSB signal and detects Cell ID.
    Main function for use from other scripts.
    
    Args:
        ssb_expected_position_ms: En fast_mode, posición esperada del SSB en ms.
                                  Si es None, procesa desde el inicio.
    """
    t0_total = time.perf_counter()

    config = get_config()

    if scs is None:
        scs = config.scs
    if sample_rate is None:
        sample_rate = config.sample_rate
    if n_symbols_display is None:
        n_symbols_display = config.n_symbols_display

    # 1) Corrección de frecuencia
    t0 = time.perf_counter()
    if verbose:
        print("Frequency correction and PSS detection...")
    search_bw = config.search_bw

    if fast_mode:
        waveform_corrected, freq_offset, nid2 = frequency_correction_ofdm_fast(
            waveform, scs, sample_rate, search_bw, verbose=verbose  # ← Añade search_bw
        )
    else:
        waveform_corrected, freq_offset, nid2 = frequency_correction_ofdm(
            waveform, scs, sample_rate, search_bw, verbose=verbose
        )
    t1 = time.perf_counter()
    if verbose:
        print(f"  → Detected NID2: {nid2}")
        print(f"  → Frequency offset: {freq_offset/1e3:.3f} kHz")

    # 2) Estimación de timing
    timing_offset = estimate_timing_offset(
        waveform_corrected, nid2, scs, sample_rate, verbose=verbose
    )
    t2 = time.perf_counter()
    waveform_aligned = waveform_corrected[timing_offset:]

    # En modo rápido, procesar solo ventana de 10ms desde el inicio
    # (después de timing_offset, el SSB está al principio)
    if fast_mode:
        max_len = int(sample_rate * 0.010)  # 10ms
        if len(waveform_aligned) > max_len:
            waveform_aligned = waveform_aligned[:max_len]

    # 3) Demodulación OFDM del primer SSB
    nrb_ssb = config.nrb_ssb
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
    t3 = time.perf_counter()

    # 4) Detección de Cell ID
    nid1, max_corr = detect_cell_id(grid_ssb, nid2, verbose=verbose)
    t4 = time.perf_counter()
    cell_id = 3 * nid1 + nid2

    # 5) Procesado de todos los SSB (omitido en fast_mode)
    if fast_mode:
        strongest_ssb = 0
        power_linear = np.mean(np.abs(grid_ssb) ** 2)
        power_db = 10 * np.log10(power_linear + 1e-12)
        pss_indices = list(range(56, 183))
        signal_power = np.mean(np.abs(grid_ssb[pss_indices, 0]) ** 2)
        noise_indices = list(range(0, 56)) + list(range(183, 240))
        noise_power = np.mean(np.abs(grid_ssb[noise_indices, :]) ** 2)
        snr_db = 10 * np.log10((signal_power / (noise_power + 1e-12)) + 1e-12)
        t5 = t4
        t6 = t4
    else:
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

        t5 = time.perf_counter()
        strongest_ssb, power_db, snr_db = detect_strongest_ssb(
            ssb_grids, nid2, nid1, lmax, verbose=verbose
        )
        t6 = time.perf_counter()

    # 6) Grid de visualización
    demod_rb = config.nrb_demod
    if fast_mode:
        # En modo rápido usa la misma ventana recortada (5 ms)
        wf_for_grid = waveform_aligned
    else:
        wf_for_grid = waveform_aligned

    grid_full = nrOFDMDemodulate(
        waveform=wf_for_grid,
        nrb=demod_rb,
        scs=scs,
        initialNSlot=0,
        SampleRate=sample_rate
    )
    t7 = time.perf_counter()

    n_symbols_available = grid_full.shape[1]
    target_symbols = min(n_symbols_display, n_symbols_available)
    grid_display = grid_full[:, :target_symbols]

    t8_total = time.perf_counter()
    total_time = (t8_total - t0_total) * 1000

    # Desglose detallado SIEMPRE para depurar
    if fast_mode:
        # En fast_mode no hay bucle de SSB bursts
        grid_full_time = (t7 - t4) * 1000
        print(
            "demodulate_ssb timings (fast): "
            f"freq_corr={(t1 - t0)*1000:.1f}ms, "
            f"timing={(t2 - t1)*1000:.1f}ms, "
            f"ssb_demod={(t3 - t2)*1000:.1f}ms, "
            f"cell_id={(t4 - t3)*1000:.1f}ms, "
            f"grid_full={grid_full_time:.1f}ms, "
            f"total={total_time:.1f}ms"
        )
    else:
        ssb_bursts_time = (t5 - t4) * 1000
        strongest_ssb_time = (t6 - t5) * 1000
        grid_full_time = (t7 - t6) * 1000
        print(
            "demodulate_ssb timings: "
            f"freq_corr={(t1 - t0)*1000:.1f}ms, "
            f"timing={(t2 - t1)*1000:.1f}ms, "
            f"ssb_demod={(t3 - t2)*1000:.1f}ms, "
            f"cell_id={(t4 - t3)*1000:.1f}ms, "
            f"ssb_bursts={ssb_bursts_time:.1f}ms, "
            f"strongest_ssb={strongest_ssb_time:.1f}ms, "
            f"grid_full={grid_full_time:.1f}ms, "
            f"total={total_time:.1f}ms"
        )


    # Calcular posición absoluta del SSB en la captura original
    ssb_absolute_position_ms = (timing_offset / sample_rate) * 1000.0
    
    return {
        'cell_id': cell_id,
        'nid1': nid1,
        'nid2': nid2,
        'strongest_ssb': strongest_ssb,
        'power_db': power_db,
        'snr_db': snr_db,
        'freq_offset': freq_offset,
        'timing_offset': timing_offset,
        'ssb_absolute_position_ms': ssb_absolute_position_ms,  # Para tracking
        'sss_correlation': max_corr,
        'grid_display': grid_display,
        'waveform_corrected': waveform_corrected
    }




def demodulate_file(mat_file: str, 
                   scs: Optional[int] = None,
                   gscn: Optional[int] = None,
                   lmax: int = 8,
                   output_folder: Optional[str] = None,
                   save_plot: bool = True,
                   save_csv: bool = False,
                   verbose: bool = False,
                   show_axes: bool = False) -> Optional[Dict[str, Any]]:
    """
    Demodulates a .mat file and optionally saves results.
    
    Args:
        mat_file: Path to .mat file
        scs: Subcarrier spacing in kHz (If None, uses config.yaml)
        gscn: Channel GSCN (If None, uses config.yaml)
        lmax: Number of SSB bursts
        output_folder: Folder to save results (None = don't save)
        save_plot: Save resource grid image
        verbose: Display detailed processing information
        show_axes: Show axes and labels in images
    
    Returns:
        Dictionary with results or None if it fails
    """
    # Cargar configuración
    config = get_config()
    if scs is None:
        scs = config.scs
    if gscn is None:
        gscn = config.gscn
    
    if verbose:
        print("="*70)
        print(f"Demodulating: {Path(mat_file).name}")
        print("="*70)
    
    try:
        # Load waveform
        waveform = load_mat_file(mat_file)
        if verbose:
            print(f"✓ Waveform loaded: {len(waveform)} samples")
        
        # Demodulate (n_symbols_display taken from config if not specified)
        results = demodulate_ssb(waveform, scs=scs, lmax=lmax, 
                                n_symbols_display=None, verbose=verbose)
        
        # Add metadata
        results['scs'] = scs
        results['sample_rate'] = 19.5e6
        results['gscn'] = gscn
        results['filename'] = Path(mat_file).name
        
        # Print results
        if verbose:
            print("\n" + "="*70)
            print("RESULTS")
            print("="*70)
            print(f"Cell ID: {results['cell_id']}")
            print(f"  NID1: {results['nid1']}")
            print(f"  NID2: {results['nid2']}")
            print(f"Strongest SSB: {results['strongest_ssb']}")
            print(f"Power: {results['power_db']:.1f} dB")
            print(f"SNR: {results['snr_db']:.1f} dB")
            print(f"Freq offset: {results['freq_offset']/1e3:.3f} kHz")
            print(f"Timing offset: {results['timing_offset']} samples")
            print("="*70)
        else:
            # Silent mode: only one line per file
            print(f"✓ {Path(mat_file).name}: Cell ID={results['cell_id']}, SNR={results['snr_db']:.1f} dB")
        
        # Save results if folder is specified
        if output_folder is not None:
            file_name = Path(mat_file).stem
            
            if save_plot and 'grid_display' in results:
                plot_resource_grid(
                    results['grid_display'],
                    results['cell_id'],
                    results['snr_db'],
                    output_folder=output_folder,
                    filename=f'{file_name}_resource_grid',
                    verbose=verbose,
                    show_axes=show_axes
                )
            
            if save_csv:
                from visualization import save_demodulation_csv
                save_demodulation_csv(results, mat_file, output_folder, f'{file_name}_data')
            
            # Only save individual log in verbose mode
            if verbose:
                save_demodulation_log(results, mat_file, output_folder, f'{file_name}_info')
        
        return results
        
    except Exception as e:
        error_msg = str(e)
        if verbose:
            print(f"\n✗ ERROR: {error_msg}")
            import traceback
            traceback.print_exc()
            # In verbose mode, save individual error log
            if output_folder is not None:
                file_name = Path(mat_file).stem
                save_error_log(e, mat_file, output_folder, f'{file_name}_ERROR')
        else:
            print(f"✗ {Path(mat_file).name}: ERROR - {error_msg[:80]}")
        
        # Return dict with error so it gets logged in the general log
        return {'error': error_msg, 'filename': Path(mat_file).name}


def demodulate_folder(folder_path: str,
                     scs: Optional[int] = None,
                     gscn: Optional[int] = None,
                     lmax: int = 8,
                     output_folder: Optional[str] = None,
                     pattern: str = "*.mat",
                     save_plot: bool = True,
                     save_csv: bool = False,
                     verbose: bool = False,
                     show_axes: bool = False,
                     num_threads: int = 4) -> Dict[str, Any]:
    """
    Demodulates all .mat files in a folder.
    
    Args:
        folder_path: Path to folder with .mat files
        scs: Subcarrier spacing in kHz (If None, uses config.yaml)
        gscn: Channel GSCN (If None, uses config.yaml)
        lmax: Number of SSB bursts
        output_folder: Folder to save results
        pattern: File pattern to process
        verbose: Display detailed processing information
        show_axes: Show axes and labels in images
    
    Returns:
        Dictionary with statistics and results
    """
    # Load configuration
    config = get_config()
    if scs is None:
        scs = config.scs
    if gscn is None:
        gscn = config.gscn
    
    folder = Path(folder_path)
    mat_files = sorted(folder.glob(pattern))
    
    if verbose:
        print(f"Found {len(mat_files)} {pattern} files in {folder_path}")
    else:
        print(f"\nProcessing {len(mat_files)} files...")
    
    # Initialize processing log
    log_file = None
    if output_folder is not None:
        log_file = init_processing_log(output_folder, len(mat_files))
    
    successful = 0
    failed = 0
    first_success = True  # To write success header only once
    first_error = True  # To write error header only once
    
    # Lock for thread-safe writing to console and logs
    print_lock = Lock()
    log_lock = Lock()
    
    def process_single_file(mat_file):
        """Helper function to process a file in a thread."""
        result = demodulate_file(
            str(mat_file),
            scs=scs,
            gscn=gscn,
            lmax=lmax,
            output_folder=output_folder,
            save_plot=save_plot,
            save_csv=save_csv,
            verbose=verbose,
            show_axes=show_axes
        )
        return mat_file, result
    
    # Parallel processing with ThreadPoolExecutor
    if num_threads > 1:
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            # Submit all jobs
            futures = {executor.submit(process_single_file, mat_file): mat_file 
                      for mat_file in mat_files}
            
            # Process results as they complete
            for future in as_completed(futures):
                mat_file, result = future.result()
                
                with log_lock:
                    if result is not None:
                        # Check if it's a successful result or an error
                        if 'error' in result:
                            failed += 1
                            if log_file is not None:
                                append_error_to_log(log_file, result['filename'], result['error'], first_error)
                                first_error = False
                        else:
                            successful += 1
                            if log_file is not None:
                                append_success_to_log(log_file, result, first_success)
                                first_success = False
                    else:
                        # Legacy case just in case (shouldn't occur)
                        failed += 1
                        if log_file is not None:
                            append_error_to_log(log_file, mat_file.name, 'Error desconocido', first_error)
                            first_error = False
                
                if verbose:
                    with print_lock:
                        print()  # Blank line between files only in verbose mode
    else:
        # Sequential processing (1 thread)
        for mat_file in mat_files:
            result = demodulate_file(
                str(mat_file),
                scs=scs,
                gscn=gscn,
                lmax=lmax,
                output_folder=output_folder,
                save_plot=save_plot,
                save_csv=save_csv,
                verbose=verbose,
                show_axes=show_axes
            )
            
            if result is not None:
                # Check if it's a successful result or an error
                if 'error' in result:
                    failed += 1
                    if log_file is not None:
                        append_error_to_log(log_file, result['filename'], result['error'], first_error)
                        first_error = False
                else:
                    successful += 1
                    if log_file is not None:
                        append_success_to_log(log_file, result, first_success)
                        first_success = False
            else:
                # Legacy case just in case (shouldn't occur)
                failed += 1
                if log_file is not None:
                    append_error_to_log(log_file, mat_file.name, 'Unknown error', first_error)
                    first_error = False
            
            if verbose:
                print()  # Blank line between files only in verbose mode
    
    print(f"\n{'='*70}")
    print(f"Processing completed: {successful} successful, {failed} failed")
    print("="*70)
    
    # Finalize processing log
    if log_file is not None:
        finalize_processing_log(log_file, successful, failed)
        print(f"✓ General log saved: {log_file}")
    
    return {
        'successful': successful,
        'failed': failed,
        'total': len(mat_files)
    }
