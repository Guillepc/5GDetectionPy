#!/usr/bin/env python3
"""
Interfaz de línea de comandos para demodulación 5G NR.
Permite procesar archivos individuales o carpetas completas.
"""

import sys
import argparse
from pathlib import Path

from nr_demodulator import demodulate_file, demodulate_folder


def main():
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
        default=30,
        choices=[15, 30],
        help='Subcarrier spacing en kHz (default: 30)'
    )
    
    parser.add_argument(
        '--gscn',
        type=int,
        default=7929,
        help='GSCN del canal (default: 7929)'
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
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    
    if not input_path.exists():
        print(f"✗ Error: No existe {args.input}")
        sys.exit(1)
    
    # Procesar archivo o carpeta
    if input_path.is_file():
        result = demodulate_file(
            str(input_path),
            scs=args.scs,
            gscn=args.gscn,
            lmax=args.lmax,
            output_folder=args.output,
            save_plot=not args.no_plot
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
            pattern=args.pattern
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
