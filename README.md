# LEDSmokeAnalysis (LEDSA)

LEDSA (LEDSmokeAnalysis) is a Python-based software package for the computation of spatially and temporally resolved light extinction coefficients from photometric measurements. The method relies on capturing the change in intensity of individual light sources due to fire-induced smoke. Images can be acquired within laboratory experiments using commercially available digital cameras.

[![PyPI](https://img.shields.io/pypi/v/ledsa)](https://pypi.org/project/ledsa/)
[![Documentation](https://github.com/FireDynamics/LEDSmokeAnalysis/actions/workflows/build_documentation.yaml/badge.svg)](https://github.com/FireDynamics/LEDSmokeAnalysis/actions/workflows/build_documentation.yaml)
[![Upload Python Package](https://github.com/FireDynamics/LEDSmokeAnalysis/actions/workflows/python-publish.yml/badge.svg)](https://github.com/FireDynamics/LEDSmokeAnalysis/actions/workflows/python-publish.yml)

## Installation

### Requirements

- Python 3.9 to 3.12
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
reference_property = sum_col_val mean_col_val
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

### Camera behind a window pane (optional)

A camera that looks into the smoke through a window pane, for example from outside the test
room, sees the LEDs along lines of sight whose part between the camera and the pane is free of
smoke. Only the part behind the pane belongs in the layers; counted from the camera, the light
paths are too long and the extinction coefficients come out too low. `pane_distance` is the
distance in m between the camera and the pane:

```ini
[experiment_geometry]
   camera_position = 0 -0.5 1.5
   pane_distance = 0.5
```

The pane is taken as normal to the line of sight, so the light path of every LED starts
`pane_distance` away from the camera. A flat pane normal to the optical axis is crossed by a line
of sight at an angle θ to the axis `pane_distance / cos θ` away from the camera, at 15° 3.5 %
farther, which is negligible as long as the pane is close to the camera compared to the LEDs. In
`config_stacked.ini`, every `[simulation_N]` section takes a `pane_distance` of its own. Without
the option, or with `None`, the light paths reach the camera as before. The camera may then stand
outside `domain_bounds`, as long as the layers hold the light paths behind the pane. Only the
geometry is corrected: a pane whose transmission changes during the test, for example by soot or
condensate, dims the LEDs like smoke. The header of each result file names the distance.

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

### Dense smoke and the read noise of the sensor

The intensities are not clipped at zero after the black level of the sensor is subtracted, so an
LED that has vanished in dense smoke can give a relative intensity at or below zero. The linear
solver leaves such LEDs out. Where the transmission reaches the read noise of the sensor, the LEDs
that remain are those whose noise happened to be positive, and the extinction coefficients come
out too low. With `sum_col_val` this hardly happens, because the pedestal of the search areas (see
below) keeps the relative intensities above zero and limits the extinction coefficients first: in
a fog test, single images came out 21 to 27 % too low at 1.24 1/m and 36 to 42 % at 1.64 1/m.
With the pedestal subtracted (`farbg_sum_col_val`) the read noise limits, by 5 to 6 % and 24 to
26 %, where the transmission of the red LEDs was 0.02 to 0.15 % of I0 and the noise of a single
image about 0.2 % of I0. Averaging seven images (`average_images`) lowered this to 13 to 17 % at
1.64 1/m.

### Pedestal from the far field of the LED strips (optional)

The sum over a search area holds light that does not dim with the LED: the dark level of the
sensor where it differs from the optical black, ambient light and light that smoke scatters into
the line of sight. In dense smoke this pedestal saturates the extinction coefficients. With

```ini
[analyse_photo]
farfield_distances = 150 300
```

in `config.ini`, step 3 estimates the pedestal for every image in two bands parallel to each LED
strip, 150 to 300 px away from it on both sides and at least 150 px away from every LED, smooths
the levels along the strip, takes the lower side and writes two more quantities:
`farbg_sum_col_val`, the sum minus the pedestal of the search area, and `farbg_pedestal`, the
pedestal itself. `reference_property = farbg_sum_col_val` analyses the corrected sum. The bands
have to lie beyond the light of the LEDs and of their halo; `farbg_pedestal / sum_col_val` of the
reference images shows whether they do, it stays below about 1 % there. An LED whose bands hold
no pixel on either side, and no neighbour on its strip within the smoothing that has one, keeps
its plain sum with a pedestal of 0. The default `None` estimates no pedestal and leaves the
result files as they are.

For the dark level of the sensor and for stray light that peaks at the strip, the far-field level
is a lower bound of the pedestal, so the corrected sum lies between the plain sum and the light
of the LED in the search area. Ambient light on a background that is brighter beside the strip
than behind the LEDs breaks this bound; it has been tested in dark rooms only. It does
not correct a search area that loses light of the LED, as behind a fogged window, which is an
error of the opposite sign.

### Features in validation

The background-corrected sums of step 3 (`bgsub_sum_col_val`, `bgsub2_sum_col_val`,
`sidebg_sum_col_val`), the weighting of the LEDs by their signal-to-noise ratio and the moving
average of the reference property, which v0.12.0 had, are left out of the releases until they are
validated. They are kept on the
[develop](https://github.com/FireDynamics/LEDSmokeAnalysis/tree/develop) branch. A config file that
still sets one of their options gets a note that the option is ignored.

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
