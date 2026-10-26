# LEDSmokeAnalysis (LEDSA)

LEDSA (LEDSmokeAnalysis) is a Python-based software package for the computation of spatially and temporally resolved light extinction coefficients from photometric measurements. The method relies on capturing the change in intensity of individual light sources due to fire-induced smoke. Images can be acquired within laboratory experiments using commercially available digital cameras.

[![Documentation](https://github.com/FireDynamics/LEDSmokeAnalysis/actions/workflows/build_documentation.yaml/badge.svg)](https://github.com/FireDynamics/LEDSmokeAnalysis/actions/workflows/build_documentation.yaml)
[![PyPI](https://github.com/FireDynamics/LEDSmokeAnalysis/actions/workflows/publish_pypi.yml/badge.svg)](https://github.com/FireDynamics/LEDSmokeAnalysis/actions/workflows/python-publish.yml)

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

The typical workflow consists of three main steps:

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

### Corrupted images

Images that cannot be read - missing, empty, truncated or stored in an unsupported
format - are skipped instead of aborting the analysis of the remaining images. Every
skipped image is reported on the console and listed in `analysis/corrupted_images.csv`,
together with the analysis step that skipped it and the reason why it could not be read:

```
image_path,analysis_step,error_type,error_message
imgs/IMG_0042.JPG,reading of the image metadata,Exiv2Error,Failed to read input data
imgs/IMG_0137.JPG,extraction of the LED intensities,OSError,image file is truncated
```

An image skipped while the metadata is read gets no entry in `image_infos.csv` and
`image_infos_analysis.csv`, so the analysis IDs of the readable images stay a gapless
sequence, just as when images are left out with `num_skip_imgs`. An image skipped later,
while the LED intensities are extracted, keeps its entry but produces no result file. It
is then left out of the extinction coefficients, and every remaining image stays paired
with its own experiment time, so the time axis of the result has a gap rather than a
shift. The stacked analysis (`-as`) leaves it out of the common time axis of its
simulations, so a time at which one camera has no intensities is dropped unless another
image of that camera lies within `time_sync_tolerance`. A restart with `-re` retries the
skipped images, which picks them up if they have been replaced in the meantime.

Three images are indispensable and therefore stop the run with a message naming the
file instead of being skipped: the reference image of step 1 (`ref_img_id`), which
defines the LED search areas, the first image of the experiment
(`first_img_experiment_id`), which defines the experiment start time, and the time
reference image.

### Background correction (optional)

By default the extinction coefficients are computed from `sum_col_val`, the plain sum of
the pixel values in the search area of each LED. Step 3 additionally writes
background-corrected sums, which can be selected with `reference_property` in
`config_analysis.ini`:

| `reference_property` | Background estimate |
|---|---|
| `sum_col_val` (default) | none |
| `bgsub_sum_col_val` | median of the pixels > 0 in a 2 px border ring of the search area, subtracted pixel-wise |
| `bgsub2_sum_col_val` | mean over the channel pixels (including 0) of the same border ring, subtracted as N·b |
| `sidebg_sum_col_val` | mean over the channel pixels in side bands next to the LED strip, subtracted as N·B |

N is the number of channel pixels in the search area. For small search areas the border
ring lies in the halo of the LED itself, whereas the side bands sample the additive
pedestal next to the LED: stray light, light from the smoke and noise clipped at 0. Under
dense smoke the box sum consists mostly of this pedestal.

The side bands run parallel to the strip, 25 to 60 px across it (`sideband_distances` in
`config.ini`); pixels close to other search areas are excluded. The direction of the strip
is estimated for each LED from its nearest neighbours in the same LED array.
`sidebg_pedestal` holds the subtracted N·B. `sidebg_sum_col_val` is NaN if the direction
cannot be estimated (an LED array with a single LED) or if the side bands contain no pixel,
for example because the image ends closer to the strip. The analysis stops with an error
naming these LEDs if the selected `reference_property` contains NaN.

`bgsub2_sum_col_val` and `sidebg_sum_col_val` are kept at least at `sideband_floor`
(default 5 %) of `sum_col_val`; values at this limit carry no information on the
extinction.

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

To introduce new, tested, documented, and stable changes, pull/merge requests into the master branch are used.

Pull request drafts can be used to communicate about changes and new functionality.

After reviewing and testing the changes, they will be merged into master.

Every merge with master is followed by introducing a new version tag corresponding to the semantic versioning paradigm.

## License

LEDSA is licensed under the MIT License. See the LICENSE file for details.

## Authors

- Kristian Börger (boerger@uni-wuppertal.de)
- Lukas Arnold (arnold@uni-wuppertal.de)
