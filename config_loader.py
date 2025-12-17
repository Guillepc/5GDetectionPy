#!/usr/bin/env python3
"""
Módulo para cargar y gestionar la configuración desde config.yaml
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional


class Config:
    """Clase para gestionar la configuración de la aplicación."""
    
    def __init__(self, config_file: str = "config.yaml"):
        """
        Inicializa la configuración desde el archivo YAML.
        
        Args:
            config_file: Ruta al archivo de configuración
        """
        self.config_file = Path(config_file)
        self._config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Carga el archivo de configuración YAML."""
        if not self.config_file.exists():
            raise FileNotFoundError(f"Archivo de configuración no encontrado: {self.config_file}")
        
        with open(self.config_file, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Obtiene un valor de configuración usando notación de punto.
        
        Args:
            key: Clave en notación de punto (ej: 'rf.gscn')
            default: Valor por defecto si no existe
            
        Returns:
            Valor de configuración o default
        """
        keys = key.split('.')
        value = self._config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    # === PROPIEDADES DE CONFIGURACIÓN RF ===
    
    @property
    def gscn(self) -> int:
        """GSCN (Global Synchronization Channel Number)."""
        return self.get('rf.gscn', 7929)
    
    @property
    def sample_rate(self) -> float:
        """Tasa de muestreo en Hz."""
        return float(self.get('rf.sample_rate', 19.5e6))
    
    @property
    def gain(self) -> float:
        """Ganancia del receptor en dB."""
        return float(self.get('rf.gain', 50))
    
    @property
    def scs(self) -> int:
        """Subcarrier spacing en kHz."""
        return self.get('rf.scs', 30)
    
    @property
    def antenna(self) -> str:
        """Antena a utilizar."""
        return self.get('rf.antenna', 'RX2')
    
    # === PROPIEDADES DE PROCESAMIENTO ===
    
    @property
    def nrb_ssb(self) -> int:
        """Número de Resource Blocks para SSB."""
        return self.get('processing.nrb_ssb', 20)
    
    @property
    def nrb_demod(self) -> int:
        """Número de Resource Blocks para demodulación."""
        return self.get('processing.nrb_demod', 45)
    
    @property
    def n_symbols_display(self) -> int:
        """Número de símbolos OFDM a demodular."""
        return self.get('processing.n_symbols_display', 14)
    
    @property
    def search_bw(self) -> float:
        """Ancho de banda de búsqueda de frecuencia en kHz."""
        return float(self.get('processing.search_bw', 90))
    
    @property
    def detection_threshold(self) -> float:
        """Umbral de detección de SSB."""
        return float(self.get('processing.detection_threshold', 1e-3))
    
    # === PROPIEDADES DE MONITOREO ===
    
    @property
    def monitor_time(self) -> float:
        """Tiempo total de monitoreo en segundos."""
        return float(self.get('monitoring.monitor_time', 0.57))
    
    @property
    def interval(self) -> float:
        """Intervalo entre capturas en segundos."""
        return float(self.get('monitoring.interval', 0.057))
    
    @property
    def frames_per_capture(self) -> int:
        """Número de frames 5G NR por captura."""
        return self.get('monitoring.frames_per_capture', 1)
    
    @property
    def save_captures(self) -> bool:
        """Guardar capturas en disco."""
        return self.get('monitoring.save_captures', False)
    
    @property
    def captures_dir(self) -> str:
        """Directorio para guardar capturas."""
        return self.get('monitoring.captures_dir', 'capturas_disco')
    
    # === PROPIEDADES DE VISUALIZACIÓN ===
    
    @property
    def enable_gui(self) -> bool:
        """Mostrar interfaz gráfica."""
        return self.get('visualization.enable_gui', True)
    
    @property
    def figure_size(self) -> tuple:
        """Tamaño de figura (ancho, alto) en pulgadas."""
        size = self.get('visualization.figure_size', [12, 8])
        return tuple(size)
    
    @property
    def colormap(self) -> str:
        """Mapa de colores para resource grid."""
        return self.get('visualization.colormap', 'jet')
    
    @property
    def interpolation(self) -> str:
        """Interpolación de imagen."""
        return self.get('visualization.interpolation', 'nearest')
    
    @property
    def verbose(self) -> bool:
        """Mostrar información detallada en consola."""
        return self.get('visualization.verbose', False)
    
    # === PROPIEDADES DE EXPORTACIÓN ===
    
    @property
    def output_dir(self) -> str:
        """Directorio de exportación."""
        return self.get('export.output_dir', 'resultados')
    
    @property
    def save_plots(self) -> bool:
        """Guardar gráficas como imágenes PNG."""
        return self.get('export.save_plots', False)
    
    @property
    def use_timestamp(self) -> bool:
        """Incluir timestamp en nombres de archivo."""
        return self.get('export.use_timestamp', True)
    
    # === PROPIEDADES DE DISPOSITIVO ===
    
    @property
    def device_index(self) -> Optional[int]:
        """Índice del dispositivo USRP."""
        return self.get('device.index')
    
    @property
    def device_serial(self) -> Optional[str]:
        """Número de serie del dispositivo USRP."""
        return self.get('device.serial')
    
    @property
    def device_args(self) -> str:
        """Argumentos adicionales del dispositivo."""
        return self.get('device.args', '')
    
    def __repr__(self) -> str:
        """Representación de la configuración."""
        return f"Config(file='{self.config_file}', gscn={self.gscn}, scs={self.scs}kHz, sr={self.sample_rate/1e6:.1f}MHz)"


# Instancia global de configuración
_config_instance: Optional[Config] = None


def load_config(config_file: str = "config.yaml") -> Config:
    """
    Carga o retorna la instancia global de configuración.
    
    Args:
        config_file: Ruta al archivo de configuración
        
    Returns:
        Instancia de Config
    """
    global _config_instance
    
    if _config_instance is None:
        _config_instance = Config(config_file)
    
    return _config_instance


def get_config() -> Config:
    """
    Obtiene la instancia global de configuración.
    Si no existe, la carga con el archivo por defecto.
    
    Returns:
        Instancia de Config
    """
    global _config_instance
    
    if _config_instance is None:
        _config_instance = Config()
    
    return _config_instance
