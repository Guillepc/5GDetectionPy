#!/usr/bin/env python3
"""
SSB Tracker - Sistema de sincronización en dos fases para captura continua.

FASE 1 (ADQUISICIÓN): Búsqueda completa del SSB (lenta, 1 vez)
FASE 2 (TRACKING): Captura sincronizada en ventana corta (rápida, continua)
"""

import numpy as np
import time
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


class SyncState(Enum):
    """Estados del sistema de sincronización."""
    ACQUISITION = "acquisition"  # Búsqueda inicial
    TRACKING = "tracking"        # Seguimiento continuo
    LOST = "lost"               # Sincronización perdida


@dataclass
class SSBTiming:
    """Información de timing del SSB."""
    ssb_start_sample: int       # Posición en muestras del inicio del SSB
    ssb_start_time_ms: float    # Posición en ms dentro del frame capturado
    frame_duration_ms: float    # Duración del frame capturado
    capture_timestamp: float    # Timestamp de captura (time.time())
    ssb_period_ms: float = 20.0 # Periodo del SSB (típicamente 20ms)
    absolute_ssb_time: Optional[float] = None  # Timestamp absoluto del SSB
    
    def __post_init__(self):
        """Calcula timestamp absoluto del SSB."""
        if self.absolute_ssb_time is None and self.capture_timestamp is not None:
            # Timestamp absoluto = timestamp captura + offset del SSB
            self.absolute_ssb_time = self.capture_timestamp + (self.ssb_start_time_ms / 1000.0)
    
    def predict_next_ssb_absolute_time(self, frames_ahead: int = 1) -> float:
        """Predice el timestamp absoluto del próximo SSB."""
        if self.absolute_ssb_time is None:
            return None
        return self.absolute_ssb_time + (frames_ahead * self.ssb_period_ms / 1000.0)
    
    def get_optimal_capture_timing(self, sample_rate: float,
                                   window_before_ms: float = 3.0,
                                   window_after_ms: float = 7.0,
                                   current_time: float = None) -> Tuple[float, float, float]:
        """
        Calcula timing óptimo de captura para alinear con próximo SSB.
        
        Args:
            sample_rate: Tasa de muestreo en Hz
            window_before_ms: Margen antes del SSB
            window_after_ms: Margen después del SSB  
            current_time: Tiempo actual (time.time())
        
        Returns:
            capture_duration: Duración de captura en segundos
            delay_until_capture: Tiempo de espera antes de capturar
            expected_ssb_offset_ms: Posición esperada del SSB en la captura (ms)
        """
        if current_time is None:
            current_time = time.time()
        
        # Calcular cuándo ocurrirá el próximo SSB
        if self.absolute_ssb_time is None:
            # Sin timing previo, capturar inmediatamente
            capture_duration = (window_before_ms + window_after_ms) / 1000.0
            return capture_duration, 0.0, window_before_ms
        
        # Tiempo desde el último SSB conocido
        time_since_last_ssb = current_time - self.absolute_ssb_time
        
        # Número de periodos SSB transcurridos
        periods_elapsed = time_since_last_ssb / (self.ssb_period_ms / 1000.0)
        
        # Próximo SSB será después de N periodos completos
        next_period = int(periods_elapsed) + 1
        next_ssb_time = self.absolute_ssb_time + (next_period * self.ssb_period_ms / 1000.0)
        
        # Momento de inicio de captura (window_before_ms antes del SSB)
        capture_start_time = next_ssb_time - (window_before_ms / 1000.0)
        
        # Delay hasta iniciar captura
        delay_until_capture = max(0.0, capture_start_time - current_time)
        
        # Duración de captura
        capture_duration = (window_before_ms + window_after_ms) / 1000.0
        
        return capture_duration, delay_until_capture, window_before_ms


class SSBTracker:
    """
    Tracker inteligente de SSB con sincronización en dos fases.
    
    ACQUISITION: Captura 20ms completos, búsqueda exhaustiva
    TRACKING: Captura ~10ms sincronizada, procesamiento rápido
    """
    
    def __init__(self, 
                 sample_rate: float,
                 ssb_period_ms: float = 20.0,
                 tracking_window_before_ms: float = 3.0,
                 tracking_window_after_ms: float = 12.0,
                 max_tracking_failures: int = 8,
                 min_snr_threshold: float = -20.0,
                 enable_multi_cell: bool = True,
                 min_confidence_for_tracking: int = 2):
        """
        Args:
            sample_rate: Tasa de muestreo en Hz
            ssb_period_ms: Periodo del SSB burst (20ms típico)
            tracking_window_before_ms: Margen antes del SSB en tracking
            tracking_window_after_ms: Margen después del SSB en tracking
            max_tracking_failures: Fallos consecutivos antes de re-sincronizar
            min_snr_threshold: SNR mínimo para considerar detección válida
            enable_multi_cell: Permitir tracking de múltiples Cell IDs
        """
        self.sample_rate = sample_rate
        self.ssb_period_ms = ssb_period_ms
        self.tracking_window_before_ms = tracking_window_before_ms
        self.tracking_window_after_ms = tracking_window_after_ms
        self.max_tracking_failures = max_tracking_failures
        self.min_snr_threshold = min_snr_threshold
        self.enable_multi_cell = enable_multi_cell
        self.min_confidence_for_tracking = min_confidence_for_tracking
        
        # Estado del tracker
        self.state = SyncState.ACQUISITION
        self.ssb_timing: Optional[SSBTiming] = None
        self.last_cfo = 0.0
        self.last_nid2 = None
        self.last_cell_id = None
        
        # Multi-celda: historial de Cell IDs detectados
        self.cell_id_history = []  # [(cell_id, count), ...]
        self.dominant_cell_id = None
        self.known_cells = set()  # Set de Cell IDs conocidos
        self.consecutive_cell_id = None  # Cell ID consecutivo para validación
        self.consecutive_count = 0  # Contador para confirmar celda
        
        # Estadísticas de tracking
        self.tracking_failures = 0
        self.total_acquisitions = 0
        self.total_tracks = 0
        self.frame_count = 0
        self.successful_frames = 0
        
        # Configuración de captura
        self._update_capture_config()
    
    def _update_capture_config(self):
        """Actualiza configuración de captura según el estado."""
        # SIEMPRE capturar 20ms completos para garantizar que hay un SSB burst
        self.capture_duration = 0.020  # 20ms
        self.capture_delay = 0.0
        self.expected_ssb_offset_ms = None
        
        if self.state == SyncState.ACQUISITION:
            # ACQUISITION: Procesamiento completo (lento pero exhaustivo)
            self.fast_mode = False
        else:  # TRACKING
            # TRACKING: Procesamiento rápido (solo primeros 5ms)
            self.fast_mode = True
    
    def get_capture_duration(self) -> float:
        """Retorna la duración de captura actual en segundos."""
        return self.capture_duration
    

    
    def should_use_fast_mode(self) -> bool:
        """Indica si usar modo rápido en demodulación."""
        return self.fast_mode and self.state == SyncState.TRACKING
    
    def get_expected_ssb_position_ms(self) -> Optional[float]:
        """Retorna posición esperada del SSB en ms (para fast_mode)."""
        if self.ssb_timing is not None:
            return self.ssb_timing.ssb_start_time_ms
        return None
    
    def get_sync_delay(self) -> float:
        """Calcula delay en segundos para sincronizar con próximo SSB burst."""
        if self.state != SyncState.TRACKING or self.ssb_timing is None:
            return 0.0
        
        current_time = time.time()
        
        # Tiempo desde última detección SSB
        time_since_ssb = current_time - self.ssb_timing.capture_timestamp
        
        # Calcular cuántos periodos completos han pasado
        ssb_period_sec = self.ssb_period_ms / 1000.0
        periods_elapsed = time_since_ssb / ssb_period_sec
        
        # Tiempo hasta próximo SSB (restar margen de 2ms para capturar antes)
        time_to_next_ssb = ((int(periods_elapsed) + 1) * ssb_period_sec) - time_since_ssb
        capture_margin = 0.002  # Empezar captura 2ms antes del SSB
        
        delay = time_to_next_ssb - capture_margin
        
        # Si el delay es negativo o muy pequeño, esperar al siguiente periodo
        if delay < 0.005:  # Menos de 5ms, ir al siguiente periodo
            delay += ssb_period_sec
        
        return max(0.0, delay)
    
    def update_timing_for_next_capture(self):
        """Actualiza configuración de captura basada en timing actual."""
        # Ya no necesario - siempre capturamos 20ms
        pass
    
    def process_demodulation_result(self, 
                                    results: Dict[str, Any],
                                    waveform_length: int,
                                    capture_timestamp: float) -> Dict[str, Any]:
        """
        Procesa resultado de demodulación y actualiza estado del tracker.
        
        Args:
            results: Diccionario con resultados de demodulate_ssb()
            waveform_length: Longitud de la señal capturada en muestras
            capture_timestamp: Timestamp de la captura (time.time())
        
        Returns:
            results enriquecido con información de tracking
        """
        self.frame_count += 1
        
        if results is None or results.get('cell_id', -1) < 0:
            # Demodulación falló
            return self._handle_failure(results)
        
        # Demodulación exitosa
        cell_id = results['cell_id']
        nid2 = results['nid2']
        freq_offset = results['freq_offset']
        timing_offset = results['timing_offset']
        
        if self.state == SyncState.ACQUISITION:
            return self._handle_acquisition_success(
                results, waveform_length, capture_timestamp,
                cell_id, nid2, freq_offset, timing_offset
            )
        else:  # TRACKING
            return self._handle_tracking_success(
                results, waveform_length, capture_timestamp,
                cell_id, nid2, freq_offset, timing_offset
            )
    
    def _update_cell_id_history(self, cell_id: int):
        """Actualiza historial de Cell IDs y determina celda dominante."""
        # Agregar al historial
        found = False
        for i, (cid, count) in enumerate(self.cell_id_history):
            if cid == cell_id:
                self.cell_id_history[i] = (cid, count + 1)
                found = True
                break
        
        if not found:
            self.cell_id_history.append((cell_id, 1))
        
        # Mantener solo últimos 10 registros
        if len(self.cell_id_history) > 10:
            self.cell_id_history.pop(0)
        
        # Determinar celda dominante (más frecuente)
        if self.cell_id_history:
            self.dominant_cell_id = max(self.cell_id_history, key=lambda x: x[1])[0]
    
    def _is_valid_detection(self, results: Dict[str, Any], strict: bool = True) -> bool:
        """Valida si la detección es confiable.
        
        Args:
            results: Resultados de demodulación
            strict: Si True, aplica validación estricta. Si False, más permisivo.
        """
        if results is None or results.get('cell_id', -1) < 0:
            return False
        
        # Verificar SNR mínimo
        snr = results.get('snr_db', -100)
        
        if strict:
            # Modo estricto: usar umbral configurado
            if snr < self.min_snr_threshold:
                return False
        else:
            # Modo permisivo: aceptar SNR más bajo en tracking
            if snr < (self.min_snr_threshold - 5.0):  # -5dB extra de margen
                return False
        
        return True
    
    def _handle_acquisition_success(self, results, waveform_length, 
                                    capture_timestamp, cell_id, nid2, 
                                    freq_offset, timing_offset):
        """Procesa adquisición exitosa."""
        # Validación más permisiva en adquisición
        if not self._is_valid_detection(results, strict=False):
            results['sync_state'] = 'acquisition_failed'
            results['tracker_info'] = (
                f"⊗ ACQUISITION: Very weak signal (SNR={results.get('snr_db', -100):.1f}dB)"
            )
            return results
        
        # Obtener posición del SSB desde resultados
        frame_duration_ms = (waveform_length / self.sample_rate) * 1000
        ssb_start_time_ms = results.get('ssb_absolute_position_ms', 0.0)
        
        # Guardar información de timing
        self.ssb_timing = SSBTiming(
            ssb_start_sample=timing_offset,
            ssb_start_time_ms=ssb_start_time_ms,
            frame_duration_ms=frame_duration_ms,
            capture_timestamp=capture_timestamp,
            ssb_period_ms=self.ssb_period_ms
        )
        
        # Guardar parámetros de la celda
        self.last_cfo = freq_offset
        self.last_nid2 = nid2
        self.last_cell_id = cell_id
        self._update_cell_id_history(cell_id)
        
        # Transición a TRACKING
        self.state = SyncState.TRACKING
        self.tracking_failures = 0
        self.total_acquisitions += 1
        self.successful_frames += 1
        self._update_capture_config()
        
        # Enriquecer resultados
        results['sync_state'] = 'acquisition'
        results['ssb_position_ms'] = ssb_start_time_ms
        results['next_capture_duration'] = self.capture_duration
        results['tracker_info'] = (
            f"✓ SYNC ACQUIRED | SSB @ {ssb_start_time_ms:.2f}ms | "
            f"Next: fast_mode (20ms capture, 5ms processing)"
        )
        
        return results
    
    def _handle_tracking_success(self, results, waveform_length,
                                 capture_timestamp, cell_id, nid2,
                                 freq_offset, timing_offset):
        """Procesa tracking exitoso."""
        # Validación menos estricta en tracking
        if not self._is_valid_detection(results, strict=False):
            self.tracking_failures += 1
            results['sync_state'] = 'tracking_weak'
            results['tracker_info'] = (
                f"⚠ WEAK: SNR={results.get('snr_db', -100):.1f}dB "
                f"({self.tracking_failures}/{self.max_tracking_failures})"
            )
            
            if self.tracking_failures >= self.max_tracking_failures:
                self._enter_acquisition_mode("Too many weak detections")
                results['sync_state'] = 'tracking_lost'
                results['tracker_info'] = "⊗ TRACKING LOST | Re-acquiring..."
            
            return results
        
        # Actualizar historial de Cell IDs
        self._update_cell_id_history(cell_id)
        
        # Verificar consistencia de Cell ID
        if self.consecutive_cell_id == cell_id:
            self.consecutive_count += 1
        else:
            self.consecutive_cell_id = cell_id
            self.consecutive_count = 1
        
        # Añadir a celdas conocidas si tiene suficiente confianza
        if self.consecutive_count >= self.min_confidence_for_tracking:
            self.known_cells.add(cell_id)
        
        # Verificar consistencia con celda conocida
        if self.enable_multi_cell:
            # Modo multi-celda: tolerar cambios entre celdas conocidas
            if cell_id not in self.known_cells and self.consecutive_count < self.min_confidence_for_tracking:
                # Celda nueva pero sin confirmar, continuar sin actualizar estado
                self.tracking_failures += 1
                results['sync_state'] = 'tracking_uncertain'
                results['tracker_info'] = (
                    f"? Unconfirmed cell: {cell_id} ({self.consecutive_count}/{self.min_confidence_for_tracking}) | "
                    f"Known: {sorted(self.known_cells)}"
                )
                
                if self.tracking_failures >= self.max_tracking_failures:
                    self._enter_acquisition_mode(f"Too many uncertain cells")
                    results['sync_state'] = 'tracking_lost'
                    results['tracker_info'] = "⊗ TRACKING LOST | Re-acquiring..."
                
                return results
            else:
                # Celda conocida o confirmada, continuar tracking
                self.last_cell_id = cell_id
        else:
            # Modo single-celda: cambio de celda → re-adquirir
            if cell_id != self.last_cell_id:
                self._enter_acquisition_mode(
                    f"Cell ID changed: {self.last_cell_id} → {cell_id}"
                )
                results['sync_state'] = 'acquisition_triggered'
                results['tracker_info'] = "⚠ Cell change detected, re-acquiring..."
                return results
        
        # Actualizar parámetros con suavizado
        alpha_cfo = 0.7
        self.last_cfo = alpha_cfo * freq_offset + (1 - alpha_cfo) * self.last_cfo
        
        # Actualizar timing desde resultados
        ssb_start_time_ms = results.get('ssb_absolute_position_ms', 0.0)
        if self.ssb_timing is not None:
            # Actualizar con medición actual (con suavizado para estabilidad)
            alpha_timing = 0.8
            self.ssb_timing.ssb_start_time_ms = (alpha_timing * ssb_start_time_ms + 
                                                  (1 - alpha_timing) * self.ssb_timing.ssb_start_time_ms)
            self.ssb_timing.ssb_start_sample = timing_offset
            self.ssb_timing.capture_timestamp = capture_timestamp
        
        # Reset contador de fallos
        self.tracking_failures = 0
        self.total_tracks += 1
        self.successful_frames += 1
        
        # Enriquecer resultados
        results['sync_state'] = 'tracking'
        results['ssb_position_ms'] = ssb_start_time_ms
        results['cfo_smoothed'] = self.last_cfo
        
        # Mostrar info de multi-celda si está habilitado
        if self.enable_multi_cell and len(self.cell_id_history) > 1:
            cell_info = f"Cells: {[c for c,_ in self.cell_id_history]}"
            results['tracker_info'] = (
                f"▶ TRACKING (#{self.total_tracks}) | Cell: {cell_id} | "
                f"{cell_info} | CFO: {self.last_cfo/1e3:.1f}kHz"
            )
        else:
            results['tracker_info'] = (
                f"▶ TRACKING (#{self.total_tracks}) | "
                f"SSB @ {ssb_start_time_ms:.2f}ms | "
                f"CFO: {self.last_cfo/1e3:.1f}kHz"
            )
        
        return results
    
    def _handle_failure(self, results):
        """Maneja fallo en demodulación."""
        self.tracking_failures += 1
        
        if results is None:
            results = {}
        
        if self.state == SyncState.ACQUISITION:
            # Fallo en adquisición → seguir intentando con frame completo
            results['sync_state'] = 'acquisition_failed'
            results['tracker_info'] = (
                f"⊗ ACQUISITION FAILED (attempt #{self.frame_count})"
            )
        else:  # TRACKING
            if self.tracking_failures >= self.max_tracking_failures:
                # Muchos fallos → re-adquirir
                self._enter_acquisition_mode(
                    f"Tracking lost after {self.tracking_failures} failures"
                )
                results['sync_state'] = 'tracking_lost'
                results['tracker_info'] = (
                    f"⊗ TRACKING LOST | Re-acquiring with full frame..."
                )
            else:
                # Fallo aislado → seguir intentando en tracking
                results['sync_state'] = 'tracking_weak'
                results['tracker_info'] = (
                    f"⚠ TRACKING WEAK ({self.tracking_failures}/"
                    f"{self.max_tracking_failures}) | Continuing..."
                )
        
        return results
    
    def _enter_acquisition_mode(self, reason: str):
        """Entra en modo adquisición."""
        print(f"\n  → Re-entering ACQUISITION mode: {reason}")
        self.state = SyncState.ACQUISITION
        self.ssb_timing = None
        self.tracking_failures = 0
        self._update_capture_config()
    
    def get_statistics(self) -> Dict[str, Any]:
        """Retorna estadísticas del tracker."""
        if self.frame_count > 0:
            tracking_efficiency = (self.total_tracks / self.frame_count) * 100
            overall_efficiency = (self.successful_frames / self.frame_count) * 100
        else:
            tracking_efficiency = 0.0
            overall_efficiency = 0.0
        
        # Calcular ratio de re-adquisiciones
        if self.total_acquisitions > 0:
            reacquisition_rate = ((self.total_acquisitions - 1) / self.frame_count) * 100
        else:
            reacquisition_rate = 0.0
        
        return {
            'state': self.state.value,
            'total_frames': self.frame_count,
            'successful_frames': self.successful_frames,
            'acquisitions': self.total_acquisitions,
            'successful_tracks': self.total_tracks,
            'tracking_efficiency': tracking_efficiency,
            'overall_efficiency': overall_efficiency,
            'reacquisition_rate': reacquisition_rate,
            'current_failures': self.tracking_failures,
            'capture_duration_ms': self.capture_duration * 1000,
            'ssb_position_ms': self.ssb_timing.ssb_start_time_ms if self.ssb_timing else None,
            'cell_id': self.last_cell_id,
            'cell_id_history': self.cell_id_history,
            'known_cells': sorted(list(self.known_cells)),
            'dominant_cell': self.dominant_cell_id,
            'cfo_khz': self.last_cfo / 1e3 if self.last_cfo else None
        }
    
    def force_reacquisition(self):
        """Fuerza re-adquisición (útil para testing o cambios manuales)."""
        self._enter_acquisition_mode("Manual reacquisition requested")
