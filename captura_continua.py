#!/usr/bin/env python3
"""
Script de captura continua 5G NR con USRP B210.
Captura señales continuamente y actualiza el resource grid en tiempo real.
"""

import uhd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import argparse
import sys
import time
from pathlib import Path
from datetime import datetime

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


class ContinuousCapture:
    """Clase para manejar captura continua y visualización."""
    
    def __init__(self, usrp, center_freq, sample_rate, gain, scs, duration, n_symbols, interval):
        self.usrp = usrp
        self.center_freq = center_freq
        self.sample_rate = sample_rate
        self.gain = gain
        self.scs = scs
        self.duration = duration
        self.n_symbols = n_symbols
        self.interval = interval
        
        # Configurar USRP
        self.usrp.set_rx_rate(sample_rate, 0)
        self.usrp.set_rx_freq(uhd.types.TuneRequest(center_freq), 0)
        self.usrp.set_rx_gain(gain, 0)
        self.usrp.set_rx_antenna("RX2", 0)
        
        # Stream
        stream_args = uhd.usrp.StreamArgs("fc32", "sc16")
        stream_args.channels = [0]
        self.rx_streamer = self.usrp.get_rx_stream(stream_args)
        
        # Buffers
        self.num_samples = int(duration * sample_rate)
        self.recv_buffer = np.zeros((1, 10000), dtype=np.complex64)
        self.metadata = uhd.types.RXMetadata()
        
        # Estadísticas
        self.capture_count = 0
        self.last_results = None
        self.capture_times = []
        
        print(f'✓ USRP configurado:')
        print(f'  Tasa de muestreo: {self.usrp.get_rx_rate(0)/1e6:.2f} MHz')
        print(f'  Frecuencia: {self.usrp.get_rx_freq(0)/1e6:.2f} MHz')
        print(f'  Ganancia: {self.usrp.get_rx_gain(0):.1f} dB')
        print(f'  Antena: {self.usrp.get_rx_antenna(0)}')
    
    def capture_one_frame(self):
        """Captura un frame de señal."""
        samples = np.zeros(self.num_samples, dtype=np.complex64)
        
        stream_cmd = uhd.types.StreamCMD(uhd.types.StreamMode.num_done)
        stream_cmd.num_samps = self.num_samples
        stream_cmd.stream_now = True
        self.rx_streamer.issue_stream_cmd(stream_cmd)
        
        capture_start_time = time.time()
        
        samples_received = 0
        while samples_received < self.num_samples:
            num_rx_samps = self.rx_streamer.recv(self.recv_buffer, self.metadata, 1.0)
            if self.metadata.error_code != uhd.types.RXMetadataErrorCode.none:
                print(f'⚠ Error en recepción: {self.metadata.strerror()}')
                break
            end_idx = min(samples_received + num_rx_samps, self.num_samples)
            samples[samples_received:end_idx] = self.recv_buffer[0, :end_idx - samples_received]
            samples_received = end_idx
        
        capture_duration = time.time() - capture_start_time
        self.capture_times.append(capture_duration)
        
        return samples, capture_duration
    
    def process_frame(self):
        """Captura y procesa un frame."""
        try:
            # Capturar
            waveform, capture_time = self.capture_one_frame()
            self.capture_count += 1
            
            # Demodular (modo rápido para captura continua)
            results = demodulate_ssb(
                waveform, 
                scs=self.scs, 
                sample_rate=self.sample_rate,
                n_symbols_display=self.n_symbols,
                verbose=False,
                fast_mode=False  # Omite procesamiento de 8 SSB bursts
            )
            
            # Verificar si se detectó SSB válido
            ssb_detected = results is not None and results.get('cell_id', -1) >= 0
            
            if ssb_detected:
                self.last_results = results
            
            # Mostrar info en consola
            timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
            
            if ssb_detected:
                print(f'[{timestamp}] Frame #{self.capture_count:3d} | '
                      f'Cap: {capture_time*1000:.1f}ms | ', end="")
                # demodulate_ssb imprime [Demod: Xms] aquí
                print(f'| ✓ SSB | '
                      f'Cell ID: {results["cell_id"]:4d} | '
                      f'SNR: {results["snr_db"]:5.1f}dB | '
                      f'Pwr: {results["power_db"]:5.1f}dB')
            else:
                print(f'[{timestamp}] Frame #{self.capture_count:3d} | '
                      f'Cap: {capture_time*1000:.1f}ms | ', end="")
                # demodulate_ssb imprime [Demod: Xms] aquí
                print(f'| ⊗ Sin SSB')
            
            return results
            
        except Exception as e:
            print(f'[{datetime.now().strftime("%H:%M:%S.%f")[:-3]}] '
                  f'Frame #{self.capture_count:3d} | '
                  f'✗ Error: {str(e)[:50]}')
            return None


def main():
    parser = argparse.ArgumentParser(
        description='Captura continua de señal 5G NR con USRP B210 y visualización en tiempo real',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Ejemplos de uso:
  %(prog)s --device-index 0
  %(prog)s --device-index 0 --gscn 7880 --interval 0.1
  %(prog)s --device-index 0 --gain 40 --duration 0.01
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
    parser.add_argument('--interval', type=float, default=0.1,
                        help='Intervalo entre capturas en segundos (default: 0.1)')
    parser.add_argument('--no-gui', action='store_true',
                        help='Sin visualización gráfica (solo logs en consola)')
    parser.add_argument('--save-images', action='store_true',
                        help='Guardar imágenes del resource grid periódicamente')
    parser.add_argument('--save-interval', type=int, default=10,
                        help='Guardar imagen cada N frames (default: 10)')
    
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
    n_symbols = config.n_symbols_display
    
    print('=== CAPTURA CONTINUA 5G NR ===\n')
    print(f'Configuración:')
    print(f'  GSCN: {gscn}')
    print(f'  SCS: {scs} kHz')
    print(f'  Ganancia: {gain} dB')
    print(f'  Sample rate: {sample_rate/1e6:.2f} MHz')
    print(f'  Duración captura: {args.duration*1000:.1f} ms')
    print(f'  Intervalo: {args.interval*1000:.1f} ms')
    
    # Calcular frecuencia central
    center_freq = gscn_to_frequency(gscn)
    print(f'  Frecuencia: {center_freq/1e6:.2f} MHz\n')
    
    try:
        # Seleccionar dispositivo
        device_args = select_usrp_device(
            device_index=args.device_index,
            device_serial=args.device_serial
        )
        
        # Crear objeto USRP
        usrp = uhd.usrp.MultiUSRP(device_args)
        
        # Crear capturador continuo
        capturer = ContinuousCapture(
            usrp=usrp,
            center_freq=center_freq,
            sample_rate=sample_rate,
            gain=gain,
            scs=scs,
            duration=args.duration,
            n_symbols=n_symbols,
            interval=args.interval
        )
        
        if args.no_gui or args.save_images:
            # Modo sin GUI - solo logs en consola (opcionalmente guardando imágenes)
            print('\n=== CAPTURA CONTINUA (Ctrl+C para detener) ===')
            if args.save_images:
                print(f'💾 Guardando imágenes cada {args.save_interval} frames en carpeta "captures/"\n')
                Path('captures').mkdir(exist_ok=True)
            else:
                print()
            
            try:
                while True:
                    results = capturer.process_frame()
                    
                    # Guardar imagen periódicamente si está habilitado
                    if args.save_images and results is not None and capturer.capture_count % args.save_interval == 0:
                        fig, ax = plt.subplots(figsize=(12, 8))
                        grid = results['grid_display']
                        im = ax.imshow(np.abs(grid), aspect='auto', cmap='jet',
                                      origin='lower', interpolation='nearest')
                        plt.colorbar(im, ax=ax, label='Magnitude')
                        ax.set_xlabel('OFDM Symbol')
                        ax.set_ylabel('Subcarrier')
                        
                        ssb_detected = results.get('cell_id', -1) >= 0
                        if ssb_detected:
                            ax.set_title(f'Frame #{capturer.capture_count} - Cell ID: {results["cell_id"]}, '
                                       f'SNR: {results["snr_db"]:.1f} dB, SSB: {results["strongest_ssb"]}')
                        else:
                            ax.set_title(f'Frame #{capturer.capture_count} - Sin SSB detectado')
                        
                        plt.tight_layout()
                        filename = f'captures/frame_{capturer.capture_count:04d}.png'
                        plt.savefig(filename, dpi=100)
                        plt.close()
                        print(f'  💾 Guardado: {filename}')
                    
                    time.sleep(args.interval)
            except KeyboardInterrupt:
                print(f'\n\n✓ Capturados {capturer.capture_count} frames')
                if capturer.capture_times:
                    avg_time = np.mean(capturer.capture_times) * 1000
                    print(f'✓ Tiempo promedio de captura: {avg_time:.2f} ms')
                if args.save_images:
                    print(f'✓ Imágenes guardadas en carpeta "captures/"')
        else:
            # Modo con GUI - visualización animada
            # Verificar si hay display disponible
            import os
            if 'DISPLAY' not in os.environ or not os.environ['DISPLAY']:
                print('\n⚠ No hay display disponible. Usa --no-gui o --save-images')
                print('  O conéctate con: ssh -X usuario@host\n')
                return
            
            # Configurar backend interactivo de matplotlib
            try:
                matplotlib.use('TkAgg')
            except ImportError:
                try:
                    matplotlib.use('Qt5Agg')
                except ImportError:
                    print('\n⚠ No se encontró backend interactivo (TkAgg o Qt5Agg)')
                    print('  Instala: sudo apt-get install python3-tk')
                    print('  O usa: --no-gui para modo sin visualización\n')
                    return
            
            print('\n=== INICIANDO VISUALIZACIÓN (Cierra ventana para detener) ===\n')
            
            # Crear figura
            fig, ax = plt.subplots(figsize=(14, 8))
            
            # Primera captura para inicializar
            results = capturer.process_frame()
            if results is None:
                print('✗ Error en primera captura')
                return
            
            grid = results['grid_display']
            im = ax.imshow(np.abs(grid), aspect='auto', cmap='jet',
                          origin='lower', interpolation='nearest',
                          vmin=0, vmax=np.percentile(np.abs(grid), 99))
            
            plt.colorbar(im, ax=ax, label='Magnitude')
            ax.set_xlabel('OFDM Symbol')
            ax.set_ylabel('Subcarrier')
            
            title_text = ax.set_title('')
            
            def update_plot(frame):
                """Función de actualización para animación."""
                results = capturer.process_frame()
                
                if results is not None:
                    grid = results['grid_display']
                    im.set_array(np.abs(grid))
                    
                    # Actualizar escala de color dinámicamente
                    vmax = np.percentile(np.abs(grid), 99)
                    im.set_clim(vmin=0, vmax=vmax)
                    
                    # Verificar si hay SSB detectado
                    ssb_detected = results.get('cell_id', -1) >= 0
                    
                    if ssb_detected:
                        # Actualizar título con SSB detectado
                        title_text.set_text(
                            f'Frame #{capturer.capture_count} - ✓ SSB DETECTADO | '
                            f'Cell ID: {results["cell_id"]} | '
                            f'SNR: {results["snr_db"]:.1f} dB | '
                            f'SSB: {results["strongest_ssb"]} | '
                            f'({center_freq/1e6:.2f} MHz)'
                        )
                    else:
                        # Grid sin SSB claro
                        title_text.set_text(
                            f'Frame #{capturer.capture_count} - ⊗ Sin SSB detectado | '
                            f'({center_freq/1e6:.2f} MHz) - Esperando próximo SSB burst...'
                        )
                
                return [im, title_text]
            
            # Crear animación
            ani = animation.FuncAnimation(
                fig, 
                update_plot,
                interval=int(args.interval * 1000),  # en milisegundos
                blit=True,
                cache_frame_data=False
            )
            
            plt.tight_layout()
            plt.show()
            
            print(f'\n✓ Capturados {capturer.capture_count} frames')
            if capturer.capture_times:
                avg_time = np.mean(capturer.capture_times) * 1000
                print(f'✓ Tiempo promedio de captura: {avg_time:.2f} ms')
        
    except KeyboardInterrupt:
        print('\n\n⚠ Interrumpido por usuario')
        sys.exit(0)
    except Exception as e:
        print(f'\n❌ Error: {e}')
        import traceback
        traceback.print_exc()
        print('\nPuedes usar --list-devices para ver dispositivos disponibles')
        sys.exit(1)


if __name__ == '__main__':
    main()
