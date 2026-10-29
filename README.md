# LEDSmokeAnalysis (LEDSA)

LEDSA (LEDSmokeAnalysis) is a Python-based software package for the computation of spatially and temporally resolved light extinction coefficients from photometric measurements. The method relies on capturing the change in intensity of individual light sources due to fire-induced smoke. Images can be acquired within laboratory experiments using commercially available digital cameras.

[![PyPI](https://img.shields.io/pypi/v/ledsa)](https://pypi.org/project/ledsa/)
[![Documentation](https://github.com/FireDynamics/LEDSmokeAnalysis/actions/workflows/build_documentation.yaml/badge.svg)](https://github.com/FireDynamics/LEDSmokeAnalysis/actions/workflows/build_documentation.yaml)
[![Upload Python Package](https://github.com/FireDynamics/LEDSmokeAnalysis/actions/workflows/python-publish.yml/badge.svg)](https://github.com/FireDynamics/LEDSmokeAnalysis/actions/workflows/python-publish.yml)

## Installation

### Requirements

- Python 3.8
- pip (Python package installer)

### Installation from PyPI

The easiest way to install LEDSA is via pip:

```bash
python -m pip install ledsa
```

### Installation from Source

To install LEDSA from source:

1. Clone the repository:
   ```bash
   git clone https://github.com/FireDynamics/LEDSmokeAnalysis.git
   cd LEDSmokeAnalysis
   ```

2. Install the package:
   ```bash
   pip install .
   ```

## Usage

LEDSA can be used via its command-line interface (CLI). The general structure is:

```bash
python -m ledsa [ARGUMENT] [OPTIONS]
```

### Configuration

Create a default configuration file:
```bash
python -m ledsa -conf
```

Create an analysis configuration file:
```bash
python -m ledsa -conf_a
```

### Main Workflow

The typical workflow:

1. **Step 1: Find LEDs on a reference image**
   ```bash
   python -m ledsa -s1
   ```

2. **Step 2: Assign LEDs to LED arrays**
   ```bash
   python -m ledsa -s2
   ```

3. **Step 3: Analyze light intensity changes among different images for the RGB color channels**

   ```bash
   python -m ledsa -s3_fast -rgb
   ```

4. **Calculate 3D coordinates of the individual LEDs**
   ```bash
   python -m ledsa -coord
   ```

5. **Run computation of extinction coefficient**
   ```bash
   python -m ledsa --analysis
   ```

For a complete list of options, run:
```bash
python -m ledsa --help
```

### Analysed quantities and result files

`reference_property` in `config_analysis.ini` names the quantity of step 3 the extinction
coefficients are computed from. Several quantities, separated by spaces, are analysed one after
another, each solver and quantity writing into a directory of its own:

```ini
[DEFAULT]
reference_property = sum_col_val bgsub_sum_col_val
```

```
analysis/extinction_coefficients/<solver>/<reference_property>/extinction_coefficients_<solver>_channel_0_<reference_property>_led_array_0.csv
```

### Reference images of the normalization

The extinction coefficients are computed from the relative intensities I/I0. The images I0 is
averaged over are named by the first of these options that is set:

| Option | File | Names the images by |
| --- | --- | --- |
| `ref_img_indices` | `config_analysis.ini` | analysis ID |
| `ref_img_experiment_ids` | `config.ini` | experiment ID, the number the file name is built from |
| `num_ref_images` | `config_analysis.ini` | the number of images at the beginning of the experiment |

Use `ref_img_experiment_ids` whenever `num_skip_imgs` or `average_images` is set: the images it
names are always analysed and never averaged. Set it before step 3.

### Stacked multi-camera analysis

Several cameras that watch the same LED arrays can be inverted jointly:

```bash
python -m ledsa -conf_s   # writes config_stacked.ini
python -m ledsa -as
```

`config_stacked.ini` holds the solver and domain settings in `[DEFAULT]`, one `[simulation_N]`
section per camera and a `[led_arrays]` section mapping virtual array IDs to the array IDs of the
individual runs. Since image IDs are counted per run, `num_ref_images` and `ref_img_indices` can
be set per `[simulation_N]` section; pick the reference images of all cameras from the same time
window before the fire.

### Background correction (optional)

Besides the plain sum `sum_col_val`, step 3 writes background-corrected sums that can be selected
as `reference_property`:

| `reference_property` | Background estimate |
|---|---|
| `sum_col_val` (default) | none |
| `bgsub_sum_col_val` | median of a 2 px border ring of the search area, subtracted pixel-wise |
| `bgsub2_sum_col_val` | mean of the same border ring, subtracted per pixel of the search area |
| `sidebg_sum_col_val` | mean of side bands next to the LED strip, subtracted per pixel of the search area |

The side bands lie `sideband_distances` (default 25 to 60 px) across the strip in `config.ini`;
raise `sideband_exclusion_margin` if light of a neighbouring strip reaches them.

### Temperature correction (optional)

LEDs lose radiant power as they heat up, which would otherwise be read as extinction. Reduction
factors determined outside LEDSA are given per camera channel in `config_analysis.ini` (and
likewise in `config_stacked.ini`):

```ini
[temperature_correction]
   reduction_file_channel_0 = reduction_factors_red.csv
   reduction_file_channel_1 = None
   reduction_file_channel_2 = None
```

The file is either a table with the time in seconds in the first column and the heights in metres
as column headers, or a list with the columns `height`, `time` and `reduction_factor`:

```
Time / s,3.3,3.2,3.1,2.9,2.6
0,1.0,1.0,1.0,1.0,1.0
60,0.991,0.993,0.994,0.996,0.998
```

The time counts from `start_time` of `config.ini`, like `Experiment_Time[s]` in
`analysis/image_infos_analysis.csv`. The intensities are divided by the bilinearly interpolated
factors before I0 is computed; images and LEDs outside the table stay uncorrected, with a
warning. The header of each result file states the correction applied.

## Demo

LEDSA includes a demo that demonstrates its functionality using sample data.

### Setting Up the Demo

The demo setup will download approximately 5GB of data from the internet:

```bash
python -m ledsa --demo --setup /path/to/demo/directory
```

This will create two directories:
- `image_data`: Contains the sample images
- `simulation`: Contains configuration files and results

### Running the Demo

After setting up the demo, you can run it:

```bash
python -m ledsa --demo --run
```

By default, the demo uses 1 core. You can specify more cores:

```bash
python -m ledsa --demo --run --n_cores 4
```

## Documentation

Comprehensive documentation is available at [https://firedynamics.github.io/LEDSmokeAnalysis/](https://firedynamics.github.io/LEDSmokeAnalysis/)

## Contributing

Changes are made via pull requests into master; drafts can be used for discussion. Every merge into master is followed by a new version tag following semantic versioning.

## License

LEDSA is licensed under the MIT License. See the LICENSE file for details.

## Authors

- Kristian Börger (boerger@uni-wuppertal.de)
- Lukas Arnold (arnold@uni-wuppertal.de)
