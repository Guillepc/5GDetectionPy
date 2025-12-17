# Demodulador 5G NR

Demodulador de señales 5G NR desarrollado en Python que detecta Cell ID, SSB, potencia y SNR desde archivos `.mat` capturados con SDR o en tiempo real con USRP B210.

## Instalación

```bash
# Crear entorno virtual
python -m venv env
source env/bin/activate  # Windows: env\Scripts\activate

# Instalar dependencias
pip install -r requirements.txt

# Para captura con USRP (opcional)
pip install uhd pyyaml
```

## Configuración

Los parámetros por defecto se definen en `config.yaml`. Este archivo centraliza todos los parámetros de RF, procesamiento, visualización y exportación. Los argumentos de la CLI sobreescriben los valores de `config.yaml`.

### Parámetros principales en config.yaml

```yaml
rf:
  gscn: 7929                 # Global Sync Channel Number
  sample_rate: 19.5e6       # Tasa de muestreo (Hz)
  scs: 30                   # Subcarrier spacing (kHz)

processing:
  nrb_ssb: 20               # Resource blocks para SSB
  nrb_demod: 45             # Resource blocks para demodulación
  search_bw: 90             # Ancho de banda de búsqueda (kHz)
```

## Uso

### 1. Procesamiento de archivos .mat (CLI)

```bash
# Procesar archivo (usa config.yaml)
python demodulate_cli.py captura.mat -o resultados/

# Sobreescribir parámetros de config.yaml
python demodulate_cli.py captura.mat -o resultados/ --scs 15 --gscn 7880

# Procesar carpeta completa
python demodulate_cli.py carpeta/ -o resultados/ --pattern "*.mat"

# Ver opciones
python demodulate_cli.py --help
```

### 2. Uso programático (API)

```python
from nr_demodulator import demodulate_file, demodulate_ssb

# Procesar archivo .mat
resultado = demodulate_file('captura.mat', output_dir='resultados/', scs=30)
print(f"Cell ID: {resultado['cell_id']}, SNR: {resultado['snr_db']:.1f} dB")

# Demodular waveform en memoria (captura en vivo)
waveform = ...  # numpy array complejo desde SDR
resultado = demodulate_ssb(waveform, scs=30, sample_rate=19.5e6, lmax=8)
```

### 3. Captura simple con USRP B210

Script simplificado para captura rápida, demodulación y visualización con ejes:

```bash
# Captura simple (usa config.yaml)
python captura_simple.py

# Con GSCN específico
python captura_simple.py --gscn 7880

# Listar dispositivos USRP
python captura_simple.py --list-devices

# Seleccionar dispositivo
python captura_simple.py --device-index 0
python captura_simple.py --device-serial 12345678

# Ajustar ganancia y duración
python captura_simple.py --gain 40 --duration 0.05
```

**Características**:
- Una sola captura (no guarda archivos)
- Visualización con ejes (X: símbolos OFDM, Y: subportadoras)
- Logs moderados (no verbosos)
- Ideal para testing rápido

### 4. Monitoreo continuo con USRP B210

Para capturas múltiples y monitoreo prolongado:

```bash
# Listar dispositivos USRP
python monitoreo_continuo.py --list-devices

# Captura y procesamiento continuo
python monitoreo_continuo.py --config config.yaml

# Modo simulación (sin hardware)
python monitoreo_continuo.py --simulate
```

**Características**:
- Múltiples capturas con intervalo configurable
- Visualización con slider temporal
- Guardado opcional de resultados
- Control completo de parámetros

## Estructura del proyecto

```
5GDetectionPy/
├── nr_demodulator.py           # API principal de demodulación
├── frequency_correction.py     # Corrección de offset de frecuencia
├── timing_estimation.py        # Estimación de timing offset
├── cell_detection.py           # Detección de Cell ID y SSB
├── visualization.py            # Visualización y logging
├── config_loader.py            # Carga de configuración YAML
├── demodulate_cli.py           # CLI para archivos .mat
├── captura_simple.py           # Captura rápida con USRP
├── monitoreo_continuo.py       # Monitoreo continuo USRP
└── config.yaml                 # Configuración centralizada
```

## Salida

Al procesar archivos se generan:

- `<archivo>_resource_grid.png` - Visualización del resource grid (540×54, 300 DPI)
- `<archivo>_info.txt` - Log con Cell ID, NID1/NID2, SNR, potencia, offsets
- `<archivo>_ERROR.txt` - Log de errores (si ocurren)

## Parámetros

Todos los scripts usan `config.yaml` por defecto. Los argumentos CLI sobreescriben estos valores.

### demodulate_cli.py

| Parámetro | Descripción | Default |
|-----------|-------------|---------|
| `--scs` | Subcarrier spacing (kHz) | config.yaml (30) |
| `--gscn` | GSCN del canal | config.yaml (7929) |
| `--lmax` | Número de SSB bursts | 8 |
| `--pattern` | Patrón de archivos | `*.mat` |
| `--verbose` | Modo detallado | False |
| `--show-axes` | Imágenes con ejes | False |
| `--no-plot` | No guardar imágenes | False |

### captura_simple.py

| Parámetro | Descripción | Default |
|-----------|-------------|---------|
| `--device-index` | Índice dispositivo USRP | Auto |
| `--device-serial` | Serial dispositivo USRP | Auto |
| `--gscn` | GSCN del canal | config.yaml (7929) |
| `--scs` | Subcarrier spacing (kHz) | config.yaml (30) |
| `--gain` | Ganancia receptor (dB) | config.yaml (50) |
| `--duration` | Duración captura (s) | 0.02 |
| `--list-devices` | Listar dispositivos USRP | - |

## Troubleshooting

```bash
# Módulos faltantes
pip install py3gpp h5py uhd pyyaml

# Verificar USRP
uhd_find_devices
python monitoreo_continuo.py --list-devices

# Sin hardware USRP
python monitoreo_continuo.py --simulate
```

## Referencias

- [py3gpp](https://github.com/NajibOdhah/py3gpp) - Implementación Python de 5G NR
- [3GPP TS 38.211](https://www.3gpp.org/DynaReport/38211.htm) - Physical channels and modulation
