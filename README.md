# Demodulador 5G NR 100% Python

Demodulador de señales 5G NR que detecta Cell ID, SSB, potencia y SNR desde archivos `.mat` capturados con SDR.

## ✅ Características

- **100% Python**: Sin dependencias de MATLAB
- **Open Source**: Usa py3gpp (implementación libre de 5G NR)
- **Cell ID correcto**: Detecta NID1 y NID2 correctamente
- **Método robusto**: Usa OFDM modulation + correlación (replica MATLAB nrTimingEstimate)
- **Validado**: Probado contra resultados de MATLAB

## 📦 Requisitos

```bash
pip install numpy scipy h5py matplotlib py3gpp
```

O usando el archivo requirements.txt:
```bash
pip install -r requirements.txt
```

## 🚀 Inicio rápido

Para ver ejemplos de uso:
```bash
python demo_usage.py
```

## 📖 Uso detallado

### Uso básico

```bash
python demodulate_5g_nr.py archivo.mat [carpeta_salida]
```

**Parámetros:**
- `archivo.mat`: Ruta al archivo .mat con la señal capturada (variable `waveform`)
- `carpeta_salida`: (Opcional) Carpeta donde guardar imagen PNG y logs TXT

### Ejemplos

**Sin guardar imágenes:**
```bash
python demodulate_5g_nr.py 5GDetection/capturas_disco_con/timestamp_20251210_120747_292.mat
```

**Guardando imágenes y logs:**
```bash
python demodulate_5g_nr.py 5GDetection/capturas_disco_con/timestamp_20251210_120747_292.mat resource_grids_output
```

**Procesamiento por lotes:**
```bash
# Procesar 5 archivos guardando imágenes
python test_batch.py 5GDetection/capturas_disco_con 5 resource_grids_batch
```

### Salida

```
======================================================================
Demodulando: timestamp_20251210_120747_292.mat
======================================================================
✓ Waveform cargado: 390000 muestras
Corrección de frecuencia y detección PSS...
  Probando 65 offsets × 3 NID2...
  → NID2: 0, Freq offset: -2.000 kHz
  Timing offset: 66911 muestras
Demodulación OFDM...
Detección de Cell ID (SSS)...
  → NID1: 0
Demodulando 8 SSB bursts...
Detección de SSB más fuerte...
  → SSB más fuerte: 0

======================================================================
RESULTADOS
======================================================================
Cell ID: 0 (NID1=0, NID2=0)
Strongest SSB: 0
Potencia: -16.3 dB
SNR: 12.4 dB
Freq offset: -2.000 kHz
Timing offset: 66911 muestras
======================================================================
```

## 🔧 Uso programático

```python
from demodulate_5g_nr import demodulate_single

result = demodulate_single(
    mat_file='archivo.mat',
    scs=30,                    # Subcarrier spacing (kHz): 15 o 30
    gscn=7929,                 # GSCN del canal (ej: 7929 para 3.75 GHz)
    lmax=8,                    # Número de SSB bursts (típicamente 8)
    verbose=True,              # Mostrar información detallada
    output_folder='mi_carpeta' # Opcional: guardar imagen y log
)

if result:
    print(f"Cell ID: {result['cell_id']}")
    print(f"NID1: {result['nid1']}, NID2: {result['nid2']}")
    print(f"Strongest SSB: {result['strongest_ssb']}")
    print(f"Potencia: {result['power_db']:.1f} dB")
    print(f"SNR: {result['snr_db']:.1f} dB")
    print(f"Freq offset: {result['freq_offset']/1e3:.3f} kHz")
    print(f"Timing offset: {result['timing_offset']} muestras")
```

### Parámetros configurables

| Parámetro | Tipo | Por defecto | Descripción |
|-----------|------|-------------|-------------|
| `mat_file` | str | - | **Requerido**. Ruta al archivo .mat |
| `scs` | int | 30 | Subcarrier spacing en kHz (15 o 30) |
| `gscn` | int | 7929 | GSCN del canal sincronización |
| `lmax` | int | 8 | Número máximo de SSB bursts |
| `verbose` | bool | True | Mostrar información detallada |
| `output_folder` | str | None | Carpeta para guardar PNG y TXT |

### Valores de retorno

La función `demodulate_single()` retorna un diccionario con:

```python
{
    'cell_id': int,           # Cell ID físico (0-1007)
    'nid1': int,              # Physical cell ID group (0-335)
    'nid2': int,              # PSS ID (0-2)
    'strongest_ssb': int,     # Índice del SSB más fuerte (0-7)
    'power_db': float,        # Potencia en dB
    'snr_db': float,          # SNR estimado en dB
    'freq_offset': float,     # Offset de frecuencia en Hz
    'timing_offset': int,     # Offset de timing en muestras
    'sss_correlation': float  # Valor de correlación SSS
}
```

## 📊 Salida de archivos

Cuando se especifica `output_folder`, el script genera:

- **`nombre_archivo_resource_grid.png`**: Imagen del resource grid con:
  - **Dimensiones**: 540 subportadoras × 54 símbolos OFDM (45 RB)
  - Mapa de calor con colormap 'jet' mostrando magnitud
  - Rectángulo blanco marcando el SSB (240 subportadoras × 4 símbolos)
  - Etiqueta del SSB más fuerte dentro del rectángulo
  - Cell ID y SNR en el título
  - **Igual formato que la versión MATLAB**
  
- **`nombre_archivo_info.txt`**: Log con información completa:
  - Cell ID, NID1, NID2
  - Strongest SSB index
  - Potencia y SNR estimados
  - Offset de frecuencia y timing
  - Parámetros de configuración (SCS, sample rate, GSCN)

- **`nombre_archivo_ERROR.txt`**: (solo si hay error) Stack trace completo

## 📊 Validación

Comparación con MATLAB para `timestamp_20251210_120747_292.mat`:

| Parámetro | Python | MATLAB | Estado |
|-----------|--------|--------|--------|
| Cell ID | 0 | 0 | ✅ |
| NID1 | 0 | 0 | ✅ |
| NID2 | 0 | 0 | ✅ |
| Freq offset | -2.0 kHz | -2.18 kHz | ✅ (~200 Hz diff) |

Probado en múltiples archivos:
- `timestamp_20251210_120747_292.mat` → Cell ID: 0 ✅
- `timestamp_20251210_120747_317.mat` → Cell ID: 0 ✅
- `timestamp_20251210_120747_384.mat` → Cell ID: 0 ✅
- `timestamp_20251210_120747_452.mat` → Cell ID: 0 ✅

## 🛠️ Detalles técnicos

### Algoritmo

1. **Corrección de frecuencia y detección PSS**:
   - Búsqueda gruesa: ±90 kHz con paso de 15 kHz
   - Búsqueda fina: ±15 kHz con paso de 500 Hz
   - Método: OFDM modulation + correlación (como MATLAB nrTimingEstimate)
   - Detecta NID2 (0, 1 o 2)

2. **Estimación de timing offset**:
   - Correlación directa con secuencia PSS
   - Encuentra inicio del SSB burst

3. **Demodulación OFDM**:
   - 4 símbolos OFDM del SSB block
   - FFT 256 puntos
   - 20 RBs (240 subportadoras)

4. **Detección de Cell ID**:
   - Extrae símbolos SSS
   - Correlaciona con 336 posibles NID1
   - Fórmula: `sum(abs(sssRx .* conj(sssRef))^2)`

5. **Detección de SSB más fuerte**:
   - Demodula 8 SSB bursts
   - Estima potencia del SSS
   - Estima SNR usando PBCH-DMRS

### Diferencias con MATLAB

- **Método PSS**: Python usa OFDM modulation explícita (más transparente)
- **Búsqueda frecuencia**: Python tiene búsqueda fina adicional
- **Precisión timing**: Python ~66911 vs MATLAB 64197 (~2700 samples = 140 µs @ 19.5 MHz)

## 📝 Formato de archivos .mat

El script soporta:
- **MATLAB v7**: Formato binario estándar
- **MATLAB v7.3**: Formato HDF5 (requiere h5py)

### Requisitos del archivo

Los archivos `.mat` deben contener:
- **Variable `waveform`**: Señal IQ compleja (muestras capturadas del SDR)
- **Formato**: Vector columna o fila (se convierte automáticamente)
- **Tipo de datos**: Complex double (real + imaginario)
- **Sample rate**: 19.5 MHz (configurable en código)

### Ejemplo de captura con SDR

```matlab
% MATLAB - Captura con SDR
rx = comm.SDRuReceiver('CenterFrequency', 3750e6, ...
                       'SampleRate', 19.5e6, ...
                       'Gain', 50, ...
                       'SamplesPerFrame', 390000);
waveform = rx();
save('captura.mat', 'waveform', '-v7.3');
```

## 🐛 Troubleshooting

### Error: "h5py no disponible"
```bash
pip install h5py
```

### Error: "No module named 'py3gpp'"
```bash
pip install py3gpp
```

### Resultados incorrectos
- Verificar que `scs` es correcto (30 kHz para FR1 banda n78)
- Verificar que el archivo .mat contiene señal 5G NR válida
- Ajustar `search_bw` si el offset de frecuencia es muy grande

## 📚 Referencias

- [py3gpp](https://github.com/NajibOdhah/py3gpp): Implementación Python de 5G NR
- [3GPP TS 38.211](https://www.3gpp.org/DynaReport/38211.htm): Physical channels and modulation
- [3GPP TS 38.213](https://www.3gpp.org/DynaReport/38213.htm): Physical layer procedures

## 👤 Autor

Desarrollo: Diciembre 2024

## 📁 Estructura del proyecto

```
5GDetectionPy/
├── demodulate_5g_nr.py      # Script principal de demodulación
├── test_batch.py             # Procesamiento por lotes
├── demo_usage.py             # Ejemplos de uso completos
├── README.md                 # Este archivo (documentación)
├── requirements.txt          # Dependencias Python
├── config.yaml               # Configuración (opcional)
└── 5GDetection/              # Carpeta de datos
    ├── capturas_disco_con/   # Capturas con señal 5G
    └── capturas_disco_sin/   # Capturas sin señal (pruebas)
```

### Archivos principales

- **`demodulate_5g_nr.py`** (16 KB): Implementación completa del demodulador
  - Funciones: `load_mat_file()`, `hssb_burst_frequency_correct_ofdm()`, `detect_cell_id_sss()`, `demodulate_single()`
  - Puede usarse como script CLI o importarse como módulo
  
- **`test_batch.py`** (2.4 KB): Procesamiento batch de múltiples archivos
  - Útil para procesar carpetas completas
  - Soporta limitación de número de archivos
  
- **`demo_usage.py`** (~2 KB): Ejemplos de uso documentados
  - `demo_basic()`: Procesamiento básico sin salida
  - `demo_with_images()`: Procesamiento con imágenes
  - `demo_multiple_files()`: Procesamiento batch

## 📄 Licencia

Este código es de uso educativo e investigación.
