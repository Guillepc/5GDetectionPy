#!/usr/bin/env python3
"""
Visualization module for 5G NR resource grids.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Backend no-GUI para uso en threads
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any
import csv


def plot_resource_grid(grid_display: np.ndarray, 
                       cell_id: int, 
                       snr_db: float,
                       output_folder: Optional[str] = None,
                       filename: str = "resource_grid",
                       show: bool = False,
                       verbose: bool = False,
                       show_axes: bool = False) -> Optional[Path]:
    """
    Generates and saves a visualization of the resource grid.
    
    Args:
        grid_display: Resource grid to visualize (subcarriers × symbols)
        cell_id: Detected Cell ID
        snr_db: Estimated SNR in dB
        output_folder: Folder where to save the image (None = don't save)
        filename: Base filename (without extension)
        show: Display the figure on screen
        verbose: Display message when saving
        show_axes: If True, shows axes and labels. Default False (no axes)
    
    Returns:
        Path of the saved file or None if not saved
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
    
    if show_axes:
        ax.set_xlabel('OFDM Symbols', fontsize=12)
        ax.set_ylabel('Subcarriers', fontsize=12)
        ax.set_title(f'Resource Grid - Cell ID: {cell_id}, SNR: {snr_db:.1f} dB', fontsize=14)
        plt.colorbar(im, ax=ax, label='Magnitude')
        
        # Grid to visualize individual resource elements
        ax.grid(True, which='both', alpha=0.2, linewidth=0.5)
        ax.set_xticks(np.arange(-0.5, grid_display.shape[1], 1), minor=True)
        ax.set_yticks(np.arange(-0.5, grid_display.shape[0], 1), minor=True)
    else:
        # No axes: only the resource grid image
        ax.axis('off')
    
    image_file = None
    if output_folder is not None:
        output_path = Path(output_folder)
        output_path.mkdir(parents=True, exist_ok=True)
        image_file = output_path / f'{filename}.png'
        plt.savefig(image_file, dpi=300, bbox_inches='tight', pad_inches=0 if not show_axes else 0.1)
        if verbose:
            print(f"✓ Image saved: {image_file}")
    
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
    Saves a text log with demodulation information.
    
    Args:
        results: Dictionary with demodulation results
        mat_file: Path of the processed .mat file
        output_folder: Folder where to save the log
        filename: Base filename (without extension)
    
    Returns:
        Path of the saved file
    """
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)
    
    log_file = output_path / f'{filename}.txt'
    with open(log_file, 'w') as f:
        f.write('=== PROCESSING INFORMATION ===\n')
        f.write(f'File: {mat_file}\n')
        f.write(f'Date: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n')
        f.write(f'Cell ID: {results["cell_id"]}\n')
        f.write(f'  NID1: {results["nid1"]}\n')
        f.write(f'  NID2: {results["nid2"]}\n')
        f.write(f'Strongest SSB: {results["strongest_ssb"]}\n')
        f.write(f'Power: {results["power_db"]:.1f} dB\n')
        f.write(f'Estimated SNR: {results["snr_db"]:.1f} dB\n')
        f.write(f'Freq offset: {results["freq_offset"]/1e3:.3f} kHz\n')
        f.write(f'Timing offset: {results["timing_offset"]} samples\n')
        if 'scs' in results:
            f.write(f'Subcarrier spacing: {results["scs"]} kHz\n')
        if 'sample_rate' in results:
            f.write(f'Sample rate: {results["sample_rate"]/1e6:.1f} MHz\n')
        if 'gscn' in results:
            f.write(f'GSCN: {results["gscn"]}\n')
    
    print(f"✓ Log saved: {log_file}")
    return log_file


def save_error_log(error: Exception, 
                   mat_file: str,
                   output_folder: str,
                   filename: str = "ERROR") -> Path:
    """
    Saves an error log with traceback information.
    
    Args:
        error: Captured exception
        mat_file: Path of the .mat file that failed
        output_folder: Folder where to save the log
        filename: Base filename (without extension)
    
    Returns:
        Path of the saved file
    """
    import traceback
    
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)
    
    error_file = output_path / f'{filename}.txt'
    with open(error_file, 'w') as f:
        f.write('=== PROCESSING ERROR ===\n')
        f.write(f'File: {mat_file}\n')
        f.write(f'Date: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n')
        f.write(f'Error: {str(error)}\n\n')
        f.write('Stack trace:\n')
        f.write(traceback.format_exc())
    
    print(f"✓ Error log saved: {error_file}")
    return error_file


def init_processing_log(output_folder: str, total_files: int,
                        filename: str = "processing_log") -> Path:
    """
    Initializes the processing log file with the header.
    
    Args:
        output_folder: Folder where to save the log
        total_files: Total number of files to process
        filename: Base filename
    
    Returns:
        Path of the log file
    """
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)
    
    log_file = output_path / f'{filename}.txt'
    with open(log_file, 'w') as f:
        f.write('=' * 70 + '\n')
        f.write('PROCESSING LOG\n')
        f.write('=' * 70 + '\n')
        f.write(f'Start date: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n')
        f.write(f'Total files to process: {total_files}\n')
        f.write('\n')
    
    return log_file


def append_success_to_log(log_file: Path, result: dict, first_success: bool = False) -> None:
    """
    Appends a successful result to the processing log.
    
    Args:
        log_file: Path of the log file
        result: Dictionary with demodulation results
        first_success: If it's the first success, writes the section header
    """
    with open(log_file, 'a') as f:
        if first_success:
            f.write('=' * 70 + '\n')
            f.write('SUCCESSFULLY PROCESSED FILES\n')
            f.write('=' * 70 + '\n\n')
        
        f.write(f"File: {result.get('filename', 'N/A')}\n")
        f.write(f"  Cell ID: {result['cell_id']}\n")
        f.write(f"  NID1: {result['nid1']}, NID2: {result['nid2']}\n")
        f.write(f"  Strongest SSB: {result['strongest_ssb']}\n")
        f.write(f"  Power: {result['power_db']:.1f} dB\n")
        f.write(f"  SNR: {result['snr_db']:.1f} dB\n")
        f.write(f"  Freq offset: {result['freq_offset']/1e3:.3f} kHz\n")
        f.write(f"  Timing offset: {result['timing_offset']} samples\n")
        f.write('\n')


def append_error_to_log(log_file: Path, filename: str, error: str,
                        first_error: bool = False) -> None:
    """
    Appends an error to the processing log.
    
    Args:
        log_file: Path of the log file
        filename: Name of the file that failed
        error: Error message
        first_error: If it's the first error, writes the section header
    """
    with open(log_file, 'a') as f:
        if first_error:
            f.write('=' * 70 + '\n')
            f.write('FILES WITH ERRORS\n')
            f.write('=' * 70 + '\n\n')
        
        f.write(f"File: {filename}\n")
        f.write(f"  Error: {error}\n")
        f.write('\n')


def finalize_processing_log(log_file: Path, successful: int, failed: int) -> Path:
    """
    Finalizes the processing log by adding the final summary.
    
    Args:
        log_file: Path of the log file
        successful: Number of successfully processed files
        failed: Number of failed files
    
    Returns:
        Path of the log file
    """
    with open(log_file, 'a') as f:
        f.write('=' * 70 + '\n')
        f.write('FINAL SUMMARY\n')
        f.write('=' * 70 + '\n')
        f.write(f'End date: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n')
        f.write(f'Total files processed: {successful + failed}\n')
        f.write(f'Successful: {successful}\n')
        f.write(f'Failed: {failed}\n')
    
    return log_file


def save_demodulation_csv(results: Dict[str, Any], 
                          mat_file: str,
                          output_folder: str,
                          filename: str = "data") -> Path:
    """
    Saves demodulation data in CSV format.
    Includes metadata and the complete resource grid.
    
    Args:
        results: Dictionary with demodulation results
        mat_file: Path of the processed .mat file
        output_folder: Folder where the CSV will be saved
        filename: Base name of the file (without extension)
    
    Returns:
        Path of the saved file
    """
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)
    
    csv_file = output_path / f'{filename}.csv'
    
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Write metadata as header
        writer.writerow(['# DEMODULATION DATA'])
        writer.writerow(['File', mat_file])
        writer.writerow(['Date', datetime.now().strftime("%Y-%m-%d %H:%M:%S")])
        writer.writerow(['Cell ID', results['cell_id']])
        writer.writerow(['NID1', results['nid1']])
        writer.writerow(['NID2', results['nid2']])
        writer.writerow(['Strongest SSB', results['strongest_ssb']])
        writer.writerow(['Power (dB)', f"{results['power_db']:.2f}"])
        writer.writerow(['SNR (dB)', f"{results['snr_db']:.2f}"])
        writer.writerow(['Freq Offset (Hz)', f"{results['freq_offset']:.2f}"])
        writer.writerow(['Timing Offset (samples)', results['timing_offset']])
        
        if 'scs' in results:
            writer.writerow(['Subcarrier Spacing (kHz)', results['scs']])
        if 'sample_rate' in results:
            writer.writerow(['Sample Rate (Hz)', results['sample_rate']])
        if 'gscn' in results:
            writer.writerow(['GSCN', results['gscn']])
        
        writer.writerow([])  # Blank line
        
        # Write resource grid if available
        if 'grid_display' in results:
            grid = results['grid_display']
            writer.writerow(['# RESOURCE GRID'])
            writer.writerow(['# Dimensions', f"{grid.shape[0]} subcarriers x {grid.shape[1]} OFDM symbols"])
            writer.writerow(['# Format', 'Each row = subcarrier, each column = OFDM symbol'])
            writer.writerow([])
            
            # Header with symbol indices
            header = ['Subcarrier'] + [f'Symbol_{i}' for i in range(grid.shape[1])]
            writer.writerow(header)
            
            # Write magnitude of each element of the resource grid
            for i in range(grid.shape[0]):
                row = [f'SC_{i}'] + [f"{np.abs(grid[i, j]):.6f}" for j in range(grid.shape[1])]
                writer.writerow(row)
    
    return csv_file
