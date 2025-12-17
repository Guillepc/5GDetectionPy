#!/usr/bin/env python3
"""
Script simple de captura y demodulación 5G NR con USRP B210.
Captura una señal, la demodula y muestra el resource grid con ejes.
"""

import uhd
import numpy as np
import matplotlib.pyplot as plt
import argparse
import sys
from pathlib import Path

from config_loader import get_config
from nr_demodulator import demodulate_ssb
from visualization import plot_resource_grid


def list_usrp_devices():
    """Lista dispositivos USRP disponibles."""
    print('\n=== DISPOSITIVOS USRP DISPONIBLES ===')
    device_addrs = uhd.find("")

    if not device_addrs:
        print('No se encontraron dispositivos USRP conectados.')
        return []

    devices = []
    for idx, addr in enumerate(device_addrs):
        print(f'\n[{idx}] Dispositivo encontrado:')
        device_info = {}
        for key in addr.keys():
            value = addr.get(key)
            print(f'    {key}: {value}')
            device_info[key] = value
        devices.append(device_info)

    print('\n' + '=' * 40)
    return devices


def select_usrp_device(device_index=None, device_serial=None):
    """Selecciona un dispositivo USRP."""
    devices = list_usrp_devices()

    if not devices:
        raise RuntimeError("No hay dispositivos USRP disponibles")

    if device_index is not None:
        if 0 <= device_index < len(devices):
            selected = devices[device_index]
            print(f'\n✓ Seleccionado dispositivo [{device_index}]: {selected.get("serial", "N/A")}')
            if 'serial' in selected:
                return f"serial={selected['serial']}"
            return ""
        else:
            raise ValueError(f"Índice {device_index} fuera de rango. Hay {len(devices)} dispositivos.")

    if device_serial is not None:
        for dev in devices:
            if dev.get('serial') == device_serial:
                print(f'\n✓ Seleccionado dispositivo con serial: {device_serial}')
                return f"serial={device_serial}"
        raise ValueError(f"No se encontró dispositivo con serial: {device_serial}")

    if len(devices) == 1:
        selected = devices[0]
        print(f'\n✓ Usando único dispositivo disponible: {selected.get("serial", "N/A")}')
        if 'serial' in selected:
            return f"serial={selected['serial']}"
        return ""

    print(f'\n⚠ Hay {len(devices)} dispositivos. Especifica --device-index o --device-serial')
    raise RuntimeError("Múltiples dispositivos encontrados. Especifica cuál usar.")


def gscn_to_frequency(gscn: int) -> float:
    """Convierte GSCN a frecuencia en Hz."""
    if 7499 <= gscn <= 22255:
        N = gscn - 7499
        freq_hz = 3000e6 + N * 1.44e6
        return freq_hz
    else:
        raise ValueError(f"GSCN {gscn} fuera de rango FR1")


def capture_waveform(center_freq, sample_rate, gain, duration, device_args=""):
    """Captura una señal con el USRP B210."""
    print('\n--- Configurando USRP B210 ---')
    
    # Crear objeto USRP
    usrp = uhd.usrp.MultiUSRP(device_args)
    
    # Configurar tasa de muestreo
    usrp.set_rx_rate(sample_rate, 0)
    actual_rate = usrp.get_rx_rate(0)
    print(f'Tasa de muestreo: {actual_rate/1e6:.2f} MHz')
    
    # Configurar frecuencia central
    tune_request = uhd.types.TuneRequest(center_freq)
    usrp.set_rx_freq(tune_request, 0)
    actual_freq = usrp.get_rx_freq(0)
    print(f'Frecuencia central: {actual_freq/1e6:.2f} MHz')
    
    # Configurar ganancia
    usrp.set_rx_gain(gain, 0)
    actual_gain = usrp.get_rx_gain(0)
    print(f'Ganancia: {actual_gain:.1f} dB')
    
    # Configurar antena
    usrp.set_rx_antenna("RX2", 0)
    print(f'Antena: {usrp.get_rx_antenna(0)}')
    
    # Capturar
    print(f'\n--- Capturando {duration*1000:.1f} ms ---')
    num_samples = int(duration * sample_rate)
    samples = np.zeros(num_samples, dtype=np.complex64)
    
    stream_args = uhd.usrp.StreamArgs("fc32", "sc16")
    stream_args.channels = [0]
    rx_streamer = usrp.get_rx_stream(stream_args)
    
    recv_buffer = np.zeros((1, 10000), dtype=np.complex64)
    metadata = uhd.types.RXMetadata()
    
    stream_cmd = uhd.types.StreamCMD(uhd.types.StreamMode.num_done)
    stream_cmd.num_samps = num_samples
    stream_cmd.stream_now = True
    rx_streamer.issue_stream_cmd(stream_cmd)
    
    samples_received = 0
    while samples_received < num_samples:
        num_rx_samps = rx_streamer.recv(recv_buffer, metadata, 1.0)
        if metadata.error_code != uhd.types.RXMetadataErrorCode.none:
            print(f'⚠ Error en recepción: {metadata.strerror()}')
            break
        end_idx = min(samples_received + num_rx_samps, num_samples)
        samples[samples_received:end_idx] = recv_buffer[0, :end_idx - samples_received]
        samples_received = end_idx
    
    print(f'✓ Capturados {len(samples)} muestras')
    power_dbm = 10 * np.log10(np.mean(np.abs(samples)**2) + 1e-12)
    print(f'✓ Potencia de señal: {power_dbm:.1f} dB')
    
    return samples


def main():
    parser = argparse.ArgumentParser(
        description='Captura simple de señal 5G NR con USRP B210',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Ejemplos de uso:
  %(prog)s
  %(prog)s --gscn 7880
  %(prog)s --device-index 0
  %(prog)s --gain 40
  %(prog)s --list-devices
        '''
    )
    
    parser.add_argument('--list-devices', action='store_true',
                        help='Listar dispositivos USRP disponibles y salir')
    parser.add_argument('--device-index', type=int, metavar='N',
                        help='Índice del dispositivo a usar (0, 1, 2, ...)')
    parser.add_argument('--device-serial', type=str, metavar='SERIAL',
                        help='Número de serie del dispositivo a usar')
    parser.add_argument('--gscn', type=int,
                        help='GSCN del canal (default: desde config.yaml)')
    parser.add_argument('--scs', type=int, choices=[15, 30],
                        help='Subcarrier spacing en kHz (default: desde config.yaml)')
    parser.add_argument('--gain', type=float,
                        help='Ganancia del receptor en dB (default: desde config.yaml)')
    parser.add_argument('--duration', type=float, default=0.02,
                        help='Duración de captura en segundos (default: 0.02)')
    
    args = parser.parse_args()
    
    # Listar dispositivos si se solicita
    if args.list_devices:
        list_usrp_devices()
        return
    
    # Cargar configuración
    config = get_config()
    
    # Usar valores de config o argumentos CLI
    gscn = args.gscn if args.gscn is not None else config.gscn
    scs = args.scs if args.scs is not None else config.scs
    gain = args.gain if args.gain is not None else config.gain
    sample_rate = config.sample_rate
    
    print('=== CAPTURA Y DEMODULACIÓN 5G NR ===\n')
    print(f'Configuración:')
    print(f'  GSCN: {gscn}')
    print(f'  SCS: {scs} kHz')
    print(f'  Ganancia: {gain} dB')
    print(f'  Sample rate: {sample_rate/1e6:.2f} MHz')
    print(f'  Duración: {args.duration*1000:.1f} ms')
    
    # Calcular frecuencia central
    center_freq = gscn_to_frequency(gscn)
    print(f'  Frecuencia: {center_freq/1e6:.2f} MHz')
    
    try:
        # Seleccionar dispositivo
        device_args = select_usrp_device(
            device_index=args.device_index,
            device_serial=args.device_serial
        )
        
        # Capturar señal
        waveform = capture_waveform(
            center_freq=center_freq,
            sample_rate=sample_rate,
            gain=gain,
            duration=args.duration,
            device_args=device_args
        )
        
        # Demodular
        print('\n--- Demodulando ---')
        n_symbols = config.n_symbols_display
        results = demodulate_ssb(waveform, scs=scs, sample_rate=sample_rate, 
                                n_symbols_display=n_symbols, verbose=False)
        
        # Mostrar resultados
        print('\n=== RESULTADOS ===')
        print(f'Cell ID: {results["cell_id"]}')
        print(f'  NID1: {results["nid1"]}')
        print(f'  NID2: {results["nid2"]}')
        print(f'Strongest SSB: {results["strongest_ssb"]}')
        print(f'Potencia: {results["power_db"]:.1f} dB')
        print(f'SNR: {results["snr_db"]:.1f} dB')
        print(f'Freq offset: {results["freq_offset"]/1e3:.3f} kHz')
        print(f'Timing offset: {results["timing_offset"]} muestras')
        
        # Visualizar resource grid con ejes
        print('\n--- Mostrando Resource Grid ---')
        fig, ax = plt.subplots(figsize=(12, 8))
        
        grid = results['grid_display']
        im = ax.imshow(np.abs(grid), aspect='auto', cmap='jet',
                      origin='lower', interpolation='nearest')
        
        plt.colorbar(im, ax=ax, label='Magnitude')
        ax.set_xlabel('OFDM Symbol')
        ax.set_ylabel('Subcarrier')
        ax.set_title(f'Resource Grid - Cell ID: {results["cell_id"]}, '
                    f'SNR: {results["snr_db"]:.1f} dB '
                    f'({center_freq/1e6:.2f} MHz)')
        
        plt.tight_layout()
        plt.show()
        
        print('\n✓ Proceso completado')
        
    except KeyboardInterrupt:
        print('\n\n⚠ Interrumpido por usuario')
        sys.exit(1)
    except Exception as e:
        print(f'\n❌ Error: {e}')
        print('\nPuedes usar --list-devices para ver dispositivos disponibles')
        sys.exit(1)


if __name__ == '__main__':
    main()
