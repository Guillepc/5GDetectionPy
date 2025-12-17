#!/usr/bin/env python3
"""
Demodulador 5G NR - Funciones principales de demodulación.
"""

import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

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
from cell_detection import detect_cell_id, detect_strongest_ssb
from visualization import (plot_resource_grid, save_demodulation_log, save_error_log,
                           init_processing_log, append_success_to_log, 
                           append_error_to_log, finalize_processing_log)
from config_loader import get_config


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
                   scs: Optional[int] = None,
                   sample_rate: Optional[float] = None,
                   lmax: int = 8,
                   n_symbols_display: Optional[int] = None,
                   verbose: bool = False) -> Dict[str, Any]:
    """
    Demodula una señal SSB y detecta Cell ID.
    Función principal para uso desde otros scripts.
    
    Args:
        waveform: Señal IQ capturada
        scs: Subcarrier spacing en kHz (15 o 30). Si None, usa config.yaml
        sample_rate: Sample rate en Hz. Si None, usa config.yaml
        lmax: Número de SSB bursts a procesar (típicamente 8)
        n_symbols_display: Número de símbolos OFDM a demodular (6-14). Si None, usa config.yaml
        verbose: Mostrar información detallada del procesamiento
    
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
    # Cargar configuración
    config = get_config()
    
    # Usar valores de config si no se especifican
    if scs is None:
        scs = config.scs
    if sample_rate is None:
        sample_rate = config.sample_rate
    if n_symbols_display is None:
        n_symbols_display = config.n_symbols_display
    
    # 1. Corrección de frecuencia
    if verbose:
        print("Corrección de frecuencia y detección PSS...")
    search_bw = config.search_bw  # En kHz desde config.yaml
    waveform_corrected, freq_offset, nid2 = frequency_correction_ofdm(
        waveform, scs, sample_rate, search_bw, verbose=verbose
    )
    if verbose:
        print(f"  → NID2 detectado: {nid2}")
        print(f"  → Offset de frecuencia: {freq_offset/1e3:.3f} kHz")
    
    # 2. Estimación de timing
    timing_offset = estimate_timing_offset(waveform_corrected, nid2, scs, sample_rate, verbose=verbose)
    waveform_aligned = waveform_corrected[timing_offset:]
    
    # 3. Demodulación OFDM del primer SSB
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
    
    # 4. Detección de Cell ID
    nid1, max_corr = detect_cell_id(grid_ssb, nid2, verbose=verbose)
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
    strongest_ssb, power_db, snr_db = detect_strongest_ssb(ssb_grids, nid2, nid1, lmax, verbose=verbose)
    
    # 7. Crear resource grid para visualización (45 RB)
    demod_rb = config.nrb_demod
    grid_full = nrOFDMDemodulate(
        waveform=waveform_aligned,
        nrb=demod_rb,
        scs=scs,
        initialNSlot=0,
        SampleRate=sample_rate
    )
    
    # Extraer solo los símbolos solicitados desde config
    n_subcarriers = grid_full.shape[0]
    n_symbols_available = grid_full.shape[1]
    target_symbols = min(n_symbols_display, n_symbols_available)
    
    # Tomar los símbolos disponibles hasta el target
    grid_display = grid_full[:, :target_symbols]
    
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
                   scs: Optional[int] = None,
                   gscn: Optional[int] = None,
                   lmax: int = 8,
                   output_folder: Optional[str] = None,
                   save_plot: bool = True,
                   save_csv: bool = False,
                   verbose: bool = False,
                   show_axes: bool = False) -> Optional[Dict[str, Any]]:
    """
    Demodula un archivo .mat y opcionalmente guarda resultados.
    
    Args:
        mat_file: Ruta al archivo .mat
        scs: Subcarrier spacing en kHz (Si None, usa config.yaml)
        gscn: GSCN del canal (Si None, usa config.yaml)
        lmax: Número de SSB bursts
        output_folder: Carpeta para guardar resultados (None = no guardar)
        save_plot: Guardar imagen del resource grid
        verbose: Mostrar información detallada del procesamiento
        show_axes: Mostrar ejes y etiquetas en las imágenes
    
    Returns:
        Diccionario con resultados o None si falla
    """
    # Cargar configuración
    config = get_config()
    if scs is None:
        scs = config.scs
    if gscn is None:
        gscn = config.gscn
    
    if verbose:
        print("="*70)
        print(f"Demodulando: {Path(mat_file).name}")
        print("="*70)
    
    try:
        # Cargar waveform
        waveform = load_mat_file(mat_file)
        if verbose:
            print(f"✓ Waveform cargado: {len(waveform)} muestras")
        
        # Demodular (n_symbols_display se toma de config si no se especifica)
        results = demodulate_ssb(waveform, scs=scs, lmax=lmax, 
                                n_symbols_display=None, verbose=verbose)
        
        # Añadir metadatos
        results['scs'] = scs
        results['sample_rate'] = 19.5e6
        results['gscn'] = gscn
        results['filename'] = Path(mat_file).name
        
        # Imprimir resultados
        if verbose:
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
        else:
            # Modo silencioso: solo una línea por archivo
            print(f"✓ {Path(mat_file).name}: Cell ID={results['cell_id']}, SNR={results['snr_db']:.1f} dB")
        
        # Guardar resultados si se especifica carpeta
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
            
            # Solo guardar log individual en modo verbose
            if verbose:
                save_demodulation_log(results, mat_file, output_folder, f'{file_name}_info')
        
        return results
        
    except Exception as e:
        error_msg = str(e)
        if verbose:
            print(f"\n✗ ERROR: {error_msg}")
            import traceback
            traceback.print_exc()
            # En modo verbose, guardar log de error individual
            if output_folder is not None:
                file_name = Path(mat_file).stem
                save_error_log(e, mat_file, output_folder, f'{file_name}_ERROR')
        else:
            print(f"✗ {Path(mat_file).name}: ERROR - {error_msg[:80]}")
        
        # Retornar dict con error para que se registre en el log general
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
    Demodula todos los archivos .mat en una carpeta.
    
    Args:
        folder_path: Ruta a la carpeta con archivos .mat
        scs: Subcarrier spacing en kHz (Si None, usa config.yaml)
        gscn: GSCN del canal (Si None, usa config.yaml)
        lmax: Número de SSB bursts
        output_folder: Carpeta para guardar resultados
        pattern: Patrón de archivos a procesar
        verbose: Mostrar información detallada del procesamiento
        show_axes: Mostrar ejes y etiquetas en las imágenes
    
    Returns:
        Diccionario con estadísticas y resultados
    """
    # Cargar configuración
    config = get_config()
    if scs is None:
        scs = config.scs
    if gscn is None:
        gscn = config.gscn
    
    folder = Path(folder_path)
    mat_files = sorted(folder.glob(pattern))
    
    if verbose:
        print(f"Encontrados {len(mat_files)} archivos {pattern} en {folder_path}")
    else:
        print(f"\nProcesando {len(mat_files)} archivos...")
    
    # Inicializar log de procesamiento
    log_file = None
    if output_folder is not None:
        log_file = init_processing_log(output_folder, len(mat_files))
    
    successful = 0
    failed = 0
    first_success = True  # Para escribir la cabecera de éxitos solo una vez
    first_error = True  # Para escribir la cabecera de errores solo una vez
    
    # Lock para escritura thread-safe en consola y logs
    print_lock = Lock()
    log_lock = Lock()
    
    def process_single_file(mat_file):
        """Función auxiliar para procesar un archivo en un thread."""
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
    
    # Procesamiento paralelo con ThreadPoolExecutor
    if num_threads > 1:
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            # Enviar todos los trabajos
            futures = {executor.submit(process_single_file, mat_file): mat_file 
                      for mat_file in mat_files}
            
            # Procesar resultados a medida que se completan
            for future in as_completed(futures):
                mat_file, result = future.result()
                
                with log_lock:
                    if result is not None:
                        # Verificar si es un resultado exitoso o un error
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
                        # Caso legacy por si acaso (no debería ocurrir)
                        failed += 1
                        if log_file is not None:
                            append_error_to_log(log_file, mat_file.name, 'Error desconocido', first_error)
                            first_error = False
                
                if verbose:
                    with print_lock:
                        print()  # Línea en blanco entre archivos solo en modo verbose
    else:
        # Procesamiento secuencial (1 thread)
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
                # Verificar si es un resultado exitoso o un error
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
                # Caso legacy por si acaso (no debería ocurrir)
                failed += 1
                if log_file is not None:
                    append_error_to_log(log_file, mat_file.name, 'Error desconocido', first_error)
                    first_error = False
            
            if verbose:
                print()  # Línea en blanco entre archivos solo en modo verbose
    
    print(f"\n{'='*70}")
    print(f"Procesamiento completado: {successful} exitosos, {failed} fallidos")
    print("="*70)
    
    # Finalizar log de procesamiento
    if log_file is not None:
        finalize_processing_log(log_file, successful, failed)
        print(f"✓ Log general guardado: {log_file}")
    
    return {
        'successful': successful,
        'failed': failed,
        'total': len(mat_files)
    }
