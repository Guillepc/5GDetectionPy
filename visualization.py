#!/usr/bin/env python3
"""
Módulo de visualización para resource grids 5G NR.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any


def plot_resource_grid(grid_display: np.ndarray, 
                       cell_id: int, 
                       snr_db: float,
                       output_folder: Optional[str] = None,
                       filename: str = "resource_grid",
                       show: bool = False) -> Optional[Path]:
    """
    Genera y guarda una visualización del resource grid.
    
    Args:
        grid_display: Resource grid a visualizar (subportadoras × símbolos)
        cell_id: Cell ID detectado
        snr_db: SNR estimado en dB
        output_folder: Carpeta donde guardar la imagen (None = no guardar)
        filename: Nombre base del archivo (sin extensión)
        show: Mostrar la figura en pantalla
    
    Returns:
        Path del archivo guardado o None si no se guardó
    """
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Interpolación 'nearest' para resource elements nítidos
    im = ax.imshow(
        np.abs(grid_display), 
        aspect='auto', 
        cmap='jet', 
        origin='lower',
        interpolation='nearest'
    )
    
    ax.set_xlabel('Símbolos OFDM', fontsize=12)
    ax.set_ylabel('Subportadoras', fontsize=12)
    ax.set_title(f'Resource Grid - Cell ID: {cell_id}, SNR: {snr_db:.1f} dB', fontsize=14)
    plt.colorbar(im, ax=ax, label='Magnitud')
    
    # Grid para visualizar resource elements individuales
    ax.grid(True, which='both', alpha=0.2, linewidth=0.5)
    ax.set_xticks(np.arange(-0.5, grid_display.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, grid_display.shape[0], 1), minor=True)
    
    image_file = None
    if output_folder is not None:
        output_path = Path(output_folder)
        output_path.mkdir(parents=True, exist_ok=True)
        image_file = output_path / f'{filename}.png'
        plt.savefig(image_file, dpi=300, bbox_inches='tight')
        print(f"✓ Imagen guardada: {image_file}")
    
    if show:
        plt.show()
    else:
        plt.close(fig)
    
    return image_file


def save_demodulation_log(results: Dict[str, Any], 
                          mat_file: str,
                          output_folder: str,
                          filename: str = "info") -> Path:
    """
    Guarda un log de texto con información de demodulación.
    
    Args:
        results: Diccionario con resultados de demodulación
        mat_file: Ruta del archivo .mat procesado
        output_folder: Carpeta donde guardar el log
        filename: Nombre base del archivo (sin extensión)
    
    Returns:
        Path del archivo guardado
    """
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)
    
    log_file = output_path / f'{filename}.txt'
    with open(log_file, 'w') as f:
        f.write('=== INFORMACIÓN DE PROCESAMIENTO ===\n')
        f.write(f'Archivo: {mat_file}\n')
        f.write(f'Fecha: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n')
        f.write(f'Cell ID: {results["cell_id"]}\n')
        f.write(f'  NID1: {results["nid1"]}\n')
        f.write(f'  NID2: {results["nid2"]}\n')
        f.write(f'Strongest SSB: {results["strongest_ssb"]}\n')
        f.write(f'Potencia: {results["power_db"]:.1f} dB\n')
        f.write(f'SNR estimado: {results["snr_db"]:.1f} dB\n')
        f.write(f'Freq offset: {results["freq_offset"]/1e3:.3f} kHz\n')
        f.write(f'Timing offset: {results["timing_offset"]} muestras\n')
        if 'scs' in results:
            f.write(f'Subcarrier spacing: {results["scs"]} kHz\n')
        if 'sample_rate' in results:
            f.write(f'Sample rate: {results["sample_rate"]/1e6:.1f} MHz\n')
        if 'gscn' in results:
            f.write(f'GSCN: {results["gscn"]}\n')
    
    print(f"✓ Log guardado: {log_file}")
    return log_file


def save_error_log(error: Exception, 
                   mat_file: str,
                   output_folder: str,
                   filename: str = "ERROR") -> Path:
    """
    Guarda un log de error con información de traceback.
    
    Args:
        error: Excepción capturada
        mat_file: Ruta del archivo .mat que falló
        output_folder: Carpeta donde guardar el log
        filename: Nombre base del archivo (sin extensión)
    
    Returns:
        Path del archivo guardado
    """
    import traceback
    
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)
    
    error_file = output_path / f'{filename}.txt'
    with open(error_file, 'w') as f:
        f.write('=== ERROR DE PROCESAMIENTO ===\n')
        f.write(f'Archivo: {mat_file}\n')
        f.write(f'Fecha: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n')
        f.write(f'Error: {str(error)}\n\n')
        f.write('Stack trace:\n')
        f.write(traceback.format_exc())
    
    print(f"✓ Log de error guardado: {error_file}")
    return error_file
