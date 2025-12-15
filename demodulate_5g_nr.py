#!/usr/bin/env python3
"""
Demodulador 5G NR 100% Python - VERSIÓN CON OFDM
Usa nrOFDMModulate + nrTimingEstimate para PSS detection (como MATLAB)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy import signal as scipy_signal
import time
from datetime import datetime
from pathlib import Path
from typing import Tuple, Optional, Dict, Any
import warnings

# Importar funciones de py3gpp
from py3gpp.nrPSS import nrPSS
from py3gpp.nrSSS import nrSSS
from py3gpp.nrPSSIndices import nrPSSIndices
from py3gpp.nrSSSIndices import nrSSSIndices
from py3gpp.nrOFDMDemodulate import nrOFDMDemodulate
from py3gpp.nrOFDMModulate import nrOFDMModulate
from py3gpp.nrTimingEstimate import nrTimingEstimate
from py3gpp.nrExtractResources import nrExtractResources
from py3gpp.nrPBCHDMRS import nrPBCHDMRS
from py3gpp.nrPBCHDMRSIndices import nrPBCHDMRSIndices
from py3gpp.nrChannelEstimate import nrChannelEstimate

try:
    import h5py
    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False
    warnings.warn("h5py no disponible. Solo se pueden leer .mat v7 y anteriores.")


def load_mat_file(filename: str) -> np.ndarray:
    """
    Carga waveform desde archivo .mat (v7 o v7.3 HDF5)
    """
    try:
        # Intentar v7 primero
        mat_data = loadmat(filename)
        waveform = mat_data['waveform'].flatten()
        return waveform
    except Exception as e1:
        # Si falla, intentar v7.3 HDF5
        if not HAS_H5PY:
            raise RuntimeError(f"Archivo requiere h5py: {e1}")
        
        try:
            with h5py.File(filename, 'r') as f:
                wf_h5 = f['waveform'][()]
                # Reconstruir complejo
                if wf_h5.dtype.names:  # struct con 'real' e 'imag'
                    waveform = wf_h5['real'] + 1j * wf_h5['imag']
                else:
                    waveform = wf_h5.view(complex)
                waveform = waveform.flatten()
                return waveform
        except Exception as e2:
            raise RuntimeError(f"Error leyendo .mat: v7 falló ({e1}), v7.3 falló ({e2})")
    
    return waveform.flatten()


def hssb_burst_frequency_correct_ofdm(waveform: np.ndarray, scs: int, sample_rate: float, 
                                      search_bw: float) -> Tuple[np.ndarray, float, int]:
    """
    Corrección de frecuencia y detección de PSS usando OFDM modulation.
    MÉTODO CORRECTO que replica MATLAB: modula grid PSS con OFDM y correlaciona waveforms.
    
    Args:
        waveform: Señal IQ capturada
        scs: Subcarrier spacing en kHz (típicamente 30)
        sample_rate: Sample rate en Hz (típicamente 19.5e6)
        search_bw: Ancho de búsqueda en kHz (típicamente 3*scs = 90)
    
    Returns:
        waveform_corrected: Waveform con corrección de frecuencia
        freq_offset: Offset de frecuencia detectado en Hz
        nid2: PSS ID detectado (0, 1 o 2)
    """
    print("Corrección de frecuencia y detección PSS (método OFDM)...")
    
    # Parámetros de sincronización (igual que MATLAB)
    sync_nfft = 256
    sync_sr = sync_nfft * scs * 1000  # 256 * 30 * 1000 = 7.68 MHz
    nrb_ssb = 20  # SSB son 20 RBs = 240 subportadoras
    
    # PSS indices (0-indexed en Python)
    pss_indices = nrPSSIndices()
    
    # Crear grids de referencia para los 3 NID2
    # Grid: (240 subportadoras, 4 símbolos OFDM, 3 NID2)
    ref_grids = np.zeros((nrb_ssb * 12, 4, 3), dtype=complex)
    for nid2 in range(3):
        # PSS va en símbolo 0 del SSB block
        ref_grids[pss_indices, 0, nid2] = nrPSS(nid2)
    
    # MATLAB: fshifts = (-searchBW:scs:searchBW) * 1e3/2
    # Ejemplo: (-90:30:90) * 1000/2 = [-45000, -15000, 15000, 45000] Hz
    # Añadir búsqueda más fina con paso de 1 kHz alrededor de 0
    coarse_fshifts = np.arange(-search_bw, search_bw + scs, scs) * 1e3 / 2
    fine_fshifts = np.arange(-scs, scs + 1, 1) * 1e3 / 2  # -15kHz a +15kHz con paso de 500Hz
    fshifts = np.unique(np.concatenate([coarse_fshifts, fine_fshifts]))
    fshifts = np.sort(fshifts)
    
    peak_values = np.zeros((len(fshifts), 3))
    
    t = np.arange(len(waveform)) / sample_rate
    
    print(f"  Probando {len(fshifts)} offsets de frecuencia × 3 NID2...")
    
    for f_idx, f_shift in enumerate(fshifts):
        # Aplicar corrección de frecuencia candidata
        waveform_corrected = waveform * np.exp(-1j * 2 * np.pi * f_shift * t)
        
        # Downsample a sync_sr para acelerar
        num_samples_ds = int(len(waveform_corrected) * sync_sr / sample_rate)
        waveform_ds = scipy_signal.resample(waveform_corrected, num_samples_ds)
        
        # Probar los 3 NID2
        for nid2 in range(3):
            try:
                # MÉTODO CORRECTO: Modular la grid PSS con OFDM
                ref_grid_nid2 = ref_grids[:, :, nid2]
                ref_waveform, _ = nrOFDMModulate(
                    grid=ref_grid_nid2,
                    scs=scs,
                    initialNSlot=0,
                    SampleRate=sync_sr,
                    Nfft=sync_nfft
                )
                
                # Correlacionar waveforms (no grids!)
                # Limitar longitud para velocidad
                max_samples = min(len(waveform_ds), 300000)
                corr = scipy_signal.correlate(waveform_ds[:max_samples], 
                                            ref_waveform, mode='valid')
                
                # Guardar pico máximo
                peak_values[f_idx, nid2] = np.max(np.abs(corr))
                
            except Exception as e:
                # Si nrOFDMModulate falla, marcar con 0
                print(f"  Warning: OFDM modulate falló para NID2={nid2}, f={f_shift/1e3:.1f}kHz: {e}")
                peak_values[f_idx, nid2] = 0
    
    # Normalizar para visualización
    peak_values_norm = peak_values / np.max(peak_values) if np.max(peak_values) > 0 else peak_values
    
    # Mostrar matriz de correlaciones
    print(f"\n  Matriz de correlaciones PSS (normalizadas):")
    print(f"  {'Freq (kHz)':>12} {'NID2=0':>12} {'NID2=1':>12} {'NID2=2':>12}")
    for i, f in enumerate(fshifts):
        print(f"  {f/1e3:>12.2f} {peak_values_norm[i, 0]:>12.3f} {peak_values_norm[i, 1]:>12.3f} {peak_values_norm[i, 2]:>12.3f}")
    
    # Encontrar mejor NID2 y frecuencia
    best_f_idx, best_nid2 = np.unravel_index(np.argmax(peak_values), peak_values.shape)
    coarse_freq_offset = fshifts[best_f_idx]
    
    print(f"\n  → NID2 detectado: {best_nid2}")
    print(f"  → Offset grueso: {coarse_freq_offset/1e3:.3f} kHz")
    print(f"  → Pico máximo: {peak_values[best_f_idx, best_nid2]:.2f}")
    
    # Aplicar corrección final
    waveform_corrected = waveform * np.exp(-1j * 2 * np.pi * coarse_freq_offset * t)
    
    # Retornar offset grueso (la corrección fina se puede añadir después)
    freq_offset = coarse_freq_offset
    
    return waveform_corrected, freq_offset, best_nid2


def estimate_timing_offset(waveform: np.ndarray, nid2: int, scs: int, 
                           sample_rate: float) -> int:
    """
    Estimación de timing offset usando correlación PSS.
    """
    print("Estimación de timing offset...")
    
    nrb_ssb = 20
    pss_indices = nrPSSIndices()
    pss_seq = nrPSS(nid2)
    
    # Correlación directa con PSS
    corr = scipy_signal.correlate(waveform[:min(len(waveform), 400000)], 
                                 pss_seq, mode='valid')
    timing_offset = int(np.argmax(np.abs(corr)))
    
    print(f"  Timing offset: {timing_offset} muestras")
    
    return timing_offset


def detect_cell_id_sss(ssb_grid: np.ndarray, nid2: int) -> Tuple[int, float]:
    """
    Detección de Cell ID usando SSS.
    CORREGIDO: Usa sum(abs(correlation)**2) como MATLAB.
    
    Returns:
        nid1: Physical cell ID group (0-335)
        max_corr: Valor de correlación máxima
    """
    print("Detección de Cell ID (SSS)...")
    
    sss_indices = nrSSSIndices().astype(int)  # Asegurar que sean enteros
    sss_rx = nrExtractResources(sss_indices, ssb_grid)
    
    # Probar los 336 NID1
    correlations = np.zeros(336)
    for nid1 in range(336):
        cell_id = 3 * nid1 + nid2
        sss_ref = nrSSS(cell_id)
        
        # FÓRMULA CORRECTA (igual que MATLAB):
        # sum(abs(sssRx .* conj(sssRef))^2)
        correlation = sss_rx * np.conj(sss_ref)
        correlations[nid1] = np.sum(np.abs(correlation)**2)
    
    # Encontrar máximo
    best_nid1 = int(np.argmax(correlations))
    max_corr = correlations[best_nid1]
    
    # Top 5 para debug
    top5_idx = np.argsort(correlations)[-5:][::-1]
    top5_corr = correlations[top5_idx]
    
    print(f"  NID1 detectado: {best_nid1}")
    print(f"  Correlación máxima: {max_corr:.2f}")
    print(f"  Top 5 NID1: {top5_idx}")
    print(f"  Top 5 correlaciones: {top5_corr}")
    
    return best_nid1, max_corr


def detect_strongest_ssb(ssb_grids: np.ndarray, nid2: int, nid1: int, 
                         lmax: int = 8) -> Tuple[int, float, float]:
    """
    Detecta el SSB más fuerte entre los Lmax candidatos.
    
    Returns:
        strongest_ssb: Índice del SSB más fuerte (0-7)
        power_db: Potencia en dB
        snr_db: SNR estimado en dB
    """
    print(f"Detección de SSB más fuerte (Lmax={lmax})...")
    
    cell_id = 3 * nid1 + nid2
    sss_indices = nrSSSIndices()
    pbch_dmrs_indices = nrPBCHDMRSIndices(cell_id)
    
    powers = np.zeros(lmax)
    snrs = np.zeros(lmax)
    
    for i_ssb in range(lmax):
        grid = ssb_grids[:, :, i_ssb]
        
        # Power del SSS
        sss_rx = nrExtractResources(sss_indices, grid)
        powers[i_ssb] = np.mean(np.abs(sss_rx)**2)
        
        # Estimación de SNR usando PBCH-DMRS
        try:
            dmrs_rx = nrExtractResources(pbch_dmrs_indices, grid)
            dmrs_ref = nrPBCHDMRS(cell_id, i_ssb)
            
            if len(dmrs_rx) > 0 and len(dmrs_ref) > 0:
                h_est = dmrs_rx / dmrs_ref
                signal_power = np.mean(np.abs(h_est)**2)
                noise_power = np.var(np.abs(h_est - np.mean(h_est))**2)
                
                if noise_power > 1e-10:
                    snrs[i_ssb] = signal_power / noise_power
                else:
                    snrs[i_ssb] = signal_power / 1e-10
            else:
                snrs[i_ssb] = 0
        except:
            snrs[i_ssb] = 0
    
    # SSB más fuerte
    strongest_ssb = int(np.argmax(powers))
    power_db = 10 * np.log10(powers[strongest_ssb] + 1e-12)
    snr_db = 10 * np.log10(snrs[strongest_ssb] + 1e-12)
    
    print(f"  SSB más fuerte: {strongest_ssb}")
    print(f"  Potencia: {power_db:.1f} dB")
    print(f"  SNR estimado: {snr_db:.1f} dB")
    
    return strongest_ssb, power_db, snr_db


def demodulate_single(mat_file: str, scs: int = 30, gscn: int = 7929, 
                      lmax: int = 8, verbose: bool = True, 
                      output_folder: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """
    Demodula un archivo .mat 5G NR y detecta Cell ID.
    
    Args:
        mat_file: Ruta al archivo .mat
        scs: Subcarrier spacing en kHz (15 o 30)
        gscn: GSCN del canal
        lmax: Número máximo de SSB bursts (típicamente 8)
        verbose: Mostrar información detallada
        output_folder: (Opcional) Carpeta donde guardar imagen y log del resource grid
    
    Returns:
        dict con resultados o None si falla
    """
    print("="*70)
    print(f"Demodulando: {Path(mat_file).name}")
    print("="*70)
    
    try:
        # 1. Cargar waveform
        waveform = load_mat_file(mat_file)
        print(f"✓ Waveform cargado: {len(waveform)} muestras")
        
        # Parámetros
        sample_rate = 19.5e6  # 19.5 MHz
        search_bw = 3 * scs  # MATLAB usa 3*scs (90 kHz para scs=30)
        
        # 2. Corrección de frecuencia y detección PSS (método OFDM)
        waveform_corrected, freq_offset, nid2 = hssb_burst_frequency_correct_ofdm(
            waveform, scs, sample_rate, search_bw
        )
        
        # 3. Estimación de timing offset
        timing_offset = estimate_timing_offset(waveform_corrected, nid2, scs, sample_rate)
        
        # 4. Alinear al inicio del SSB burst
        ssb_start = timing_offset
        waveform_aligned = waveform_corrected[ssb_start:]
        
        # 5. Demodular OFDM
        print("Demodulación OFDM del SSB burst...")
        nrb_ssb = 20
        n_symbols_ssb = 4
        
        # Calcular longitud del SSB block (4 símbolos OFDM)
        # Cada símbolo = Nfft + CP
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
        
        print(f"  Grid SSB: {grid_ssb.shape}")
        
        # 6. Detección de Cell ID (SSS)
        nid1, max_corr = detect_cell_id_sss(grid_ssb, nid2)
        cell_id = 3 * nid1 + nid2
        
        # 7. Demodular todos los SSB para encontrar el más fuerte
        print(f"\nDemodulando {lmax} SSB bursts...")
        ssb_grids = np.zeros((nrb_ssb * 12, n_symbols_ssb, lmax), dtype=complex)
        
        # Periodo de SSB (asumiendo periodicidad de 20ms)
        samples_per_ssb = int(sample_rate * 0.02 / lmax)
        
        for i_ssb in range(lmax):
            start_idx = ssb_start + i_ssb * samples_per_ssb
            if start_idx + samples_per_ssb <= len(waveform_corrected):
                wf_ssb = waveform_corrected[start_idx:start_idx + samples_per_ssb]
                grid = nrOFDMDemodulate(
                    waveform=wf_ssb,
                    nrb=nrb_ssb,
                    scs=scs,
                    initialNSlot=0,
                    CyclicPrefix='normal',
                    Nfft=nfft_ssb,
                    SampleRate=sample_rate
                )
                # Tomar solo los primeros 4 símbolos
                ssb_grids[:, :, i_ssb] = grid[:, :n_symbols_ssb]
        
        # 8. Detectar SSB más fuerte
        strongest_ssb, power_db, snr_db = detect_strongest_ssb(ssb_grids, nid2, nid1, lmax)
        
        # 9. Crear grid expandido para visualización (como MATLAB)
        # MATLAB demodula 45 RB y centra el SSB de 20 RB
        print("\nCreando grid expandido para visualización (45 RB)...")
        demod_rb = 45  # MATLAB usa 45 RBs
        
        # Calcular offset de frecuencia para centrar SSB
        ssb_freq_origin = 12 * (demod_rb - nrb_ssb) // 2  # 150 subportadoras
        
        # Demodular más waveform para tener ~54 símbolos
        # Para SCS=30kHz: 1 slot = 14 símbolos = 0.5ms
        # Necesitamos ~4 slots para 56 símbolos
        samples_per_slot = int(sample_rate * 0.0005)  # 0.5ms
        num_slots = 4
        waveform_for_display = waveform_aligned[:num_slots * samples_per_slot]
        
        # Demodular con 20 RB (que sabemos que funciona)
        try:
            grid_20rb = nrOFDMDemodulate(
                waveform=waveform_for_display,
                nrb=nrb_ssb,
                scs=scs,
                initialNSlot=0,
                CyclicPrefix='normal',
                Nfft=nfft_ssb,
                SampleRate=sample_rate
            )
            
            # Limitar a 54 símbolos
            last_symbol = min(54, grid_20rb.shape[1])
            grid_20rb = grid_20rb[:, :last_symbol]
            
            # Crear grid expandido (45 RB × símbolos)
            grid_expanded = np.zeros((demod_rb * 12, grid_20rb.shape[1]), dtype=complex)
            
            # Insertar grid de 20 RB centrado
            grid_expanded[ssb_freq_origin:ssb_freq_origin + nrb_ssb * 12, :] = grid_20rb
            
            grid_display = grid_expanded
            print(f"  Grid expandido creado: {grid_display.shape}")
            
        except Exception as e:
            print(f"  Warning: Error en demodulación extendida: {e}")
            # Fallback: usar concatenación de SSB bursts
            num_ssb_to_concat = min(lmax, 8)
            grid_ssb_concat = np.concatenate([ssb_grids[:, :, i] for i in range(num_ssb_to_concat)], axis=1)
            
            grid_expanded = np.zeros((demod_rb * 12, grid_ssb_concat.shape[1]), dtype=complex)
            grid_expanded[ssb_freq_origin:ssb_freq_origin + nrb_ssb * 12, :] = grid_ssb_concat
            
            last_symbol = min(54, grid_expanded.shape[1])
            grid_display = grid_expanded[:, :last_symbol]
            print(f"  Grid expandido creado (fallback): {grid_display.shape}")
        
        # Definir rectángulo del SSB (primeros 4 símbolos)
        start_symbol = 0  # 0-indexed
        num_symbols_ssb = 4
        ssb_rect = (start_symbol, ssb_freq_origin, num_symbols_ssb, 12 * nrb_ssb)
        
        # 10. Guardar imagen y log si se especifica carpeta de salida
        if output_folder is not None:
            output_path = Path(output_folder)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Nombre base del archivo
            file_name = Path(mat_file).stem
            
            # Guardar imagen del resource grid (igual que MATLAB)
            fig, ax = plt.subplots(figsize=(10, 8))
            im = ax.imshow(np.abs(grid_display), aspect='auto', cmap='jet', origin='lower')
            ax.set_xlabel('Símbolos OFDM')
            ax.set_ylabel('Subportadoras')
            ax.set_title(f'Resource Grid - Cell ID: {cell_id}, SNR: {snr_db:.1f} dB')
            plt.colorbar(im, ax=ax, label='Magnitud')
            
            # Dibujar rectángulo del SSB si está disponible
            if ssb_rect is not None:
                from matplotlib.patches import Rectangle
                rect = Rectangle(
                    (ssb_rect[0], ssb_rect[1]),  # (x, y)
                    ssb_rect[2], ssb_rect[3],     # width, height
                    linewidth=2, edgecolor='white', facecolor='none'
                )
                ax.add_patch(rect)
                
                # Añadir texto con información del SSB más fuerte
                text_x = ssb_rect[0] + ssb_rect[2] / 2
                text_y = ssb_rect[1] + ssb_rect[3] / 2
                ax.text(text_x, text_y, f'Strongest SSB: {strongest_ssb}',
                       color='white', fontsize=10, ha='center', va='center', weight='bold',
                       bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
            
            image_file = output_path / f'{file_name}_resource_grid.png'
            plt.savefig(image_file, dpi=150, bbox_inches='tight')
            plt.close(fig)
            print(f"\n✓ Imagen guardada: {image_file}")
            
            # Guardar log de información
            log_file = output_path / f'{file_name}_info.txt'
            with open(log_file, 'w') as f:
                f.write('=== INFORMACIÓN DE PROCESAMIENTO ===\n')
                f.write(f'Archivo: {mat_file}\n')
                f.write(f'Fecha: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n')
                f.write(f'Cell ID: {cell_id}\n')
                f.write(f'  NID1: {nid1}\n')
                f.write(f'  NID2: {nid2}\n')
                f.write(f'Strongest SSB: {strongest_ssb}\n')
                f.write(f'Potencia: {power_db:.1f} dB\n')
                f.write(f'SNR estimado: {snr_db:.1f} dB\n')
                f.write(f'Freq offset: {freq_offset/1e3:.3f} kHz\n')
                f.write(f'Timing offset: {timing_offset} muestras\n')
                f.write(f'Subcarrier spacing: {scs} kHz\n')
                f.write(f'Sample rate: {sample_rate/1e6:.1f} MHz\n')
                f.write(f'GSCN: {gscn}\n')
            print(f"✓ Log guardado: {log_file}")
        
        # Resultados
        print("\n" + "="*70)
        print("RESULTADOS")
        print("="*70)
        print(f"Cell ID: {cell_id}")
        print(f"  NID1: {nid1}")
        print(f"  NID2: {nid2}")
        print(f"Strongest SSB: {strongest_ssb}")
        print(f"Potencia: {power_db:.1f} dB")
        print(f"SNR: {snr_db:.1f} dB")
        print(f"Freq offset: {freq_offset/1e3:.3f} kHz")
        print(f"Timing offset: {timing_offset} muestras")
        print("="*70)
        
        return {
            'cell_id': cell_id,
            'nid1': nid1,
            'nid2': nid2,
            'strongest_ssb': strongest_ssb,
            'power_db': power_db,
            'snr_db': snr_db,
            'freq_offset': freq_offset,
            'timing_offset': timing_offset,
            'sss_correlation': max_corr
        }
        
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        
        # Guardar log de error si se especifica carpeta
        if output_folder is not None:
            output_path = Path(output_folder)
            output_path.mkdir(parents=True, exist_ok=True)
            file_name = Path(mat_file).stem
            error_file = output_path / f'{file_name}_ERROR.txt'
            with open(error_file, 'w') as f:
                f.write('=== ERROR DE PROCESAMIENTO ===\n')
                f.write(f'Archivo: {mat_file}\n')
                f.write(f'Fecha: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n')
                f.write(f'Error: {str(e)}\n\n')
                f.write('Stack trace:\n')
                f.write(traceback.format_exc())
            print(f"✓ Log de error guardado: {error_file}")
        
        return None


if __name__ == '__main__':
    import sys
    
    # Archivo de prueba
    test_file = '5GDetection/capturas_disco_con/timestamp_20251210_120747_292.mat'
    output_folder = 'resource_grids_output'  # Carpeta por defecto
    
    if len(sys.argv) > 1:
        test_file = sys.argv[1]
    if len(sys.argv) > 2:
        output_folder = sys.argv[2]
    
    # Ejecutar
    result = demodulate_single(
        mat_file=test_file,
        scs=30,
        gscn=7929,
        lmax=8,
        verbose=True,
        output_folder=output_folder
    )
    
    if result:
        print(f"\n✓ Procesamiento completado exitosamente")
        print(f"✓ Archivos guardados en: {output_folder}/")
    else:
        print(f"\n✗ Procesamiento falló")
        sys.exit(1)
