# 5G NR Demodulator

5G NR signal demodulator developed in Python that detects Cell ID, SSB, power and SNR from `.mat` files captured with SDR or in real-time with USRP B210.

## Installation

```bash
# Create virtual environment
python -m venv env
source env/bin/activate  # Windows: env\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# For USRP capture (optional)
pip install uhd pyyaml
```

## Configuration

Default parameters are defined in `config.yaml`. This file centralizes all RF, processing, visualization and export parameters. CLI arguments override values from `config.yaml`.

### Main parameters in config.yaml

```yaml
rf:
  gscn: 7929                 # Global Sync Channel Number
  sample_rate: 19.5e6       # Sample rate (Hz)
  scs: 30                   # Subcarrier spacing (kHz)

processing:
  nrb_ssb: 20               # Resource blocks for SSB
  nrb_demod: 45             # Resource blocks for demodulation
  search_bw: 90             # Search bandwidth (kHz)
```

## Usage

### 1. Processing .mat files (CLI)

#### Basic usage

```bash
# Process individual file
python demodulate_cli.py captura.mat -o resultados/

# Process complete folder
python demodulate_cli.py carpeta/ -o resultados/

# Override config.yaml parameters
python demodulate_cli.py captura.mat -o resultados/ --scs 15 --gscn 7880

# View all options
python demodulate_cli.py --help
```

#### Parallel processing (multiple threads)

The CLI uses **parallel processing by default** (4 threads) to optimize processing time for multiple files:

```bash
# Use 4 threads (default, ~40% faster than sequential)
python demodulate_cli.py carpeta/ -o resultados/

# Use 8 threads for large folders
python demodulate_cli.py carpeta/ -o resultados/ --threads 8

# Sequential processing (1 thread)
python demodulate_cli.py carpeta/ -o resultados/ --threads 1
```

**Note**: Parallel processing is only activated when processing folders from the CLI. When importing functions from another Python script (`demodulate_file`, `demodulate_ssb`), processing is always sequential.

#### Export formats

Control which files are generated with `--export`:

```bash
# Only resource grid PNG images (default)
python demodulate_cli.py carpeta/ -o resultados/

# Only CSV files with demodulated data
python demodulate_cli.py carpeta/ -o resultados/ --export csv

# Images and CSV simultaneously
python demodulate_cli.py carpeta/ -o resultados/ --export both
```

**Available formats**:
- `images` (default): Generates `<archivo>_resource_grid.png` (540×54 visualization, 300 DPI)
- `csv`: Generates `<archivo>_data.csv` with:
  - Metadata: Cell ID, NID1/NID2, SNR, power, offsets
  - Complete resource grid: magnitude of each subcarrier × OFDM symbol
  - Compatible with Excel, pandas, MATLAB
- `both`: Generates PNG + CSV

#### Advanced options

```bash
# Specific file pattern
python demodulate_cli.py carpeta/ -o resultados/ --pattern "timestamp_*.mat"

# Verbose mode (detailed logs per file)
python demodulate_cli.py carpeta/ -o resultados/ --verbose

# Images with axes and labels
python demodulate_cli.py carpeta/ -o resultados/ --show-axes

# No images (equivalent to --export csv)
python demodulate_cli.py carpeta/ -o resultados/ --no-plot
```

#### Complete example

```bash
# Optimized processing: 8 threads, CSV + images, specific pattern
python demodulate_cli.py test_samples/presence/ \
    -o resultados_presencia/ \
    --export both \
    --threads 8 \
    --pattern "timestamp_*.mat" \
    --scs 30 \
    --gscn 7929
```

### 2. CSV to Image Converter

Convert previously generated CSV files back to resource grid images:

```bash
# Convert single CSV file
python csv_to_image.py results/file_data.csv -o images/

# Convert entire folder with 4 threads (default)
python csv_to_image.py results/ -o images/

# Convert with 8 threads for faster processing
python csv_to_image.py results/ -o images/ --threads 8

# Generate images with axes and labels
python csv_to_image.py results/ -o images/ --show-axes

# Verbose mode (detailed information)
python csv_to_image.py results/ -o images/ --verbose

# View all options
python csv_to_image.py --help
```

**Features**:
- Parallel processing (default: 4 threads)
- Reads metadata and resource grid from CSV
- Generates identical images to original demodulation
- Images without axes by default (use `--show-axes` to add them)

### 3. Programmatic usage (API)

```python
from nr_demodulator import demodulate_file, demodulate_ssb

# Process .mat file
result = demodulate_file('capture.mat', output_dir='results/', scs=30)
print(f"Cell ID: {result['cell_id']}, SNR: {result['snr_db']:.1f} dB")

# Demodulate waveform in memory (live capture)
waveform = ...  # complex numpy array from SDR
result = demodulate_ssb(waveform, scs=30, sample_rate=19.5e6, lmax=8)
```

### 4. Simple capture with USRP B210

Simplified script for quick capture, demodulation and visualization with axes:

```bash
# Simple capture (uses config.yaml)
python captura_simple.py

# With specific GSCN
python captura_simple.py --gscn 7880

# List USRP devices
python captura_simple.py --list-devices

# Select device
python captura_simple.py --device-index 0
python captura_simple.py --device-serial 12345678

# Adjust gain and duration
python captura_simple.py --gain 40 --duration 0.05
```

**Features**:
- Single capture (doesn't save files)
- Visualization with axes (X: OFDM symbols, Y: subcarriers)
- Moderate logs (not verbose)
- Ideal for quick testing

### 5. Continuous monitoring with USRP B210

For multiple captures and prolonged monitoring:

```bash
# List USRP devices
python monitoreo_continuo.py --list-devices

# Continuous capture and processing
python monitoreo_continuo.py --config config.yaml

# Simulation mode (no hardware)
python monitoreo_continuo.py --simulate
```

**Features**:
- Multiple captures with configurable interval
- Visualization with temporal slider
- Optional result saving
- Complete parameter control

## Project structure

```
5GDetectionPy/
├── nr_demodulator.py           # Main demodulation API
├── frequency_correction.py     # Frequency offset correction
├── timing_estimation.py        # Timing offset estimation
├── cell_detection.py           # Cell ID and SSB detection
├── visualization.py            # Visualization and logging
├── config_loader.py            # YAML configuration loader
├── demodulate_cli.py           # CLI for .mat files
├── csv_to_image.py             # CSV to image converter
├── captura_simple.py           # Quick capture with USRP
├── monitoreo_continuo.py       # Continuous USRP monitoring
└── config.yaml                 # Centralized configuration
```

## Output

When processing files, the following are generated (depending on options):

### Files generated with `--export images` (default)
- `<archivo>_resource_grid.png` - Resource grid visualization (540×54, 300 DPI)
- `<archivo>_info.txt` - Individual log (only in `--verbose` mode)
- `<archivo>_ERROR.txt` - Error log (if errors occur in `--verbose` mode)
- `processing_log.txt` - Complete summary with all processed files

### Files generated with `--export csv`
- `<archivo>_data.csv` - Demodulated data in CSV:
  - Metadata: Cell ID, NID1, NID2, SNR, power, offsets, etc.
  - Complete resource grid: magnitude of each subcarrier × OFDM symbol
  - Compatible with Excel, pandas, MATLAB, etc.
- `processing_log.txt` - Complete processing summary

### Files generated with `--export both`
- All previous files (PNG + CSV + logs)

## Parameters

All scripts use `config.yaml` by default. CLI arguments override these values.

### demodulate_cli.py

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--scs` | Subcarrier spacing (kHz: 15 or 30) | config.yaml (30) |
| `--gscn` | Channel GSCN | config.yaml (7929) |
| `--lmax` | Number of SSB bursts | 8 |
| `--pattern` | File pattern for folders | `*.mat` |
| `--threads` | Number of threads for parallel processing | 4 |
| `--export` | Output format: `images`, `csv`, `both` | `images` |
| `--verbose` | Detailed mode with individual logs | False |
| `--show-axes` | Images with axes and labels | False |
| `--no-plot` | Don't save images (deprecated, use `--export csv`) | False |
| `--profile` | Enable profiling to measure execution times | False |
| `--profile-output` | Save profiling result to file | None |

### csv_to_image.py

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--pattern` | File pattern for folders | `*.csv` |
| `--threads` | Number of threads for parallel processing | 4 |
| `--show-axes` | Images with axes and labels | False |
| `--verbose` | Detailed mode with processing information | False |

### captura_simple.py

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--device-index` | USRP device index | Auto |
| `--device-serial` | USRP device serial | Auto |
| `--gscn` | Channel GSCN | config.yaml (7929) |
| `--scs` | Subcarrier spacing (kHz) | config.yaml (30) |
| `--gain` | Receiver gain (dB) | config.yaml (50) |
| `--duration` | Capture duration (s) | 0.02 |
| `--list-devices` | List USRP devices | - |

## Profiling and Performance Measurement

To analyze performance and execution times:

### Profiling with .mat files

```bash
# Basic profiling (display in console)
python demodulate_cli.py file.mat -o results --profile

# Save profiling to file
python demodulate_cli.py file.mat -o results --profile --profile-output profile.stats

# Analyze profiling file
python -m pstats profile.stats
> sort cumulative
> stats 20
```

### Profiling with folders

```bash
# Profile folder processing with multiple threads
python demodulate_cli.py folder/ -o results --profile --threads 8

# Profile with specific export format
python demodulate_cli.py folder/ -o results --profile --export both
```

**Measured stages**:
1. Frequency correction (PSS correlation)
2. Timing offset estimation
3. OFDM demodulation (SSB)
4. Cell ID detection (SSS correlation)
5. Complete OFDM demodulation
6. Strongest SSB detection

## Troubleshooting

```bash
# Missing modules
pip install py3gpp h5py uhd pyyaml

# Verify USRP
uhd_find_devices
python monitoreo_continuo.py --list-devices

# Without USRP hardware
python monitoreo_continuo.py --simulate
```

## References

- [py3gpp](https://github.com/NajibOdhah/py3gpp) - Python implementation of 5G NR
- [3GPP TS 38.211](https://www.3gpp.org/DynaReport/38211.htm) - Physical channels and modulation
