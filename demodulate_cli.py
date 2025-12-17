#!/usr/bin/env python3
"""
Interfaz de línea de comandos para demodulación 5G NR.
Permite procesar archivos individuales o carpetas completas.
"""

import sys
import argparse
from pathlib import Path

from nr_demodulator import demodulate_file, demodulate_folder
from config_loader import get_config


def main():
    # Cargar configuración para defaults
    config = get_config()
    
    parser = argparse.ArgumentParser(
        description='Demodulador 5G NR - Procesa archivos .mat individuales o carpetas completas'
    )
    
    parser.add_argument(
        'input',
        type=str,
        help='Archivo .mat o carpeta con archivos .mat'
    )
    
    parser.add_argument(
        '-o', '--output',
        type=str,
        default='demodulation_results',
        help='Carpeta de salida para resultados (default: demodulation_results)'
    )
    
    parser.add_argument(
        '--scs',
        type=int,
        default=None,
        choices=[15, 30],
        help=f'Subcarrier spacing en kHz (default: {config.scs} desde config.yaml)'
    )
    
    parser.add_argument(
        '--gscn',
        type=int,
        default=None,
        help=f'GSCN del canal (default: {config.gscn} desde config.yaml)'
    )
    
    parser.add_argument(
        '--lmax',
        type=int,
        default=8,
        help='Número de SSB bursts (default: 8)'
    )
    
    parser.add_argument(
        '--pattern',
        type=str,
        default='*.mat',
        help='Patrón de archivos para carpetas (default: *.mat)'
    )
    
    parser.add_argument(
        '--no-plot',
        action='store_true',
        help='No guardar imágenes de resource grids'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Mostrar información detallada del procesamiento (por defecto modo silencioso)'
    )
    
    parser.add_argument(
        '--show-axes',
        action='store_true',
        help='Mostrar ejes y etiquetas en las imágenes (por defecto sin ejes)'
    )
    
    parser.add_argument(
        '--threads',
        type=int,
        default=4,
        help='Número de threads para procesamiento paralelo (default: 4)'
    )
    
    parser.add_argument(
        '--export',
        type=str,
        default='images',
        choices=['images', 'csv', 'both'],
        help='Formato de exportación: images (resource grids), csv (datos demodulados), o both (default: images)'
    )
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    
    if not input_path.exists():
        print(f"✗ Error: No existe {args.input}")
        sys.exit(1)
    
    # Determinar si guardar imágenes o CSV según --export
    save_plot = (args.export in ['images', 'both']) and not args.no_plot
    save_csv = (args.export in ['csv', 'both'])
    
    # Procesar archivo o carpeta
    if input_path.is_file():
        result = demodulate_file(
            str(input_path),
            scs=args.scs,
            gscn=args.gscn,
            lmax=args.lmax,
            output_folder=args.output,
            save_plot=save_plot,
            save_csv=save_csv,
            verbose=args.verbose,
            show_axes=args.show_axes
        )
        
        if result:
            print(f"\n✓ Procesamiento completado exitosamente")
            print(f"✓ Resultados guardados en: {args.output}/")
            sys.exit(0)
        else:
            print(f"\n✗ Procesamiento falló")
            sys.exit(1)
    
    elif input_path.is_dir():
        summary = demodulate_folder(
            str(input_path),
            scs=args.scs,
            gscn=args.gscn,
            lmax=args.lmax,
            output_folder=args.output,
            pattern=args.pattern,
            save_plot=save_plot,
            save_csv=save_csv,
            verbose=args.verbose,
            show_axes=args.show_axes,
            num_threads=args.threads
        )
        
        if summary['successful'] > 0:
            print(f"\n✓ Resultados guardados en: {args.output}/")
            sys.exit(0 if summary['failed'] == 0 else 2)
        else:
            print(f"\n✗ Todos los archivos fallaron")
            sys.exit(1)
    
    else:
        print(f"✗ Error: {args.input} no es un archivo o carpeta válido")
        sys.exit(1)


if __name__ == '__main__':
    main()
