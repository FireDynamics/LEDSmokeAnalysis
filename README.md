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

### Analysed quantities and result files

`reference_property` in `config_analysis.ini` names the quantity of step 3 the extinction
coefficients are computed from (see [Background correction](#background-correction-optional) for
the available ones). Several quantities, separated by spaces, are analysed one after another in a
single run:

```ini
[DEFAULT]
reference_property = sum_col_val bgsub_sum_col_val sidebg_sum_col_val
```

Every solver and every quantity writes into a directory of its own, so that the results of one
run stay apart from each other:

```
analysis/
└── extinction_coefficients/
    ├── linear/
    │   ├── sum_col_val/
    │   │   └── extinction_coefficients_linear_channel_0_sum_col_val_led_array_0.csv
    │   └── bgsub_sum_col_val/
    │       └── extinction_coefficients_linear_channel_0_bgsub_sum_col_val_led_array_0.csv
    └── nonlinear/
        └── sum_col_val/
            └── extinction_coefficients_nonlinear_channel_0_sum_col_val_led_array_0.csv
```

The stacked analysis reads its quantities from `reference_property` in `config_stacked.ini` and
writes them into the same layout below its `output_path`. Result files of earlier LEDSA versions
lie directly in `analysis/extinction_coefficients/<solver>/` and are left where they are.

### Reference images of the normalization

The extinction coefficients are computed from the relative intensities I/I0, so the images I0 is
averaged over decide what the analysis takes for clear air. Three options name them, and the
first one that is set wins:

| Option | File | Names the images by |
| --- | --- | --- |
| `ref_img_indices` | `config_analysis.ini` | analysis ID |
| `ref_img_experiment_ids` | `config.ini` | experiment ID, the number the file name is built from |
| `num_ref_images` | `config_analysis.ini` | the number of images at the beginning of the experiment |

`ref_img_experiment_ids` is the only one the data extraction reads, so it is the one to use
whenever `num_skip_imgs` or `average_images` is set:

```ini
[analyse_photo]
num_skip_imgs = 9
ref_img_experiment_ids = 385 386 387 388 389 390
```

The images it names are analysed whatever `num_skip_imgs` leaves out, so the reference window
keeps the time span it is meant to have instead of being pulled apart by the increment, and
`average_images` keeps every one of them as it is instead of summarizing it with its neighbours.
They have to lie between the first and the last analysed image, and they have to be named
before step 3, since an image that was never analysed holds no intensities.

The analysis prints the image IDs I0 was really computed from, while the header of the result
file names the reference images as the configuration gave them. With `average_images` the two
differ by the renumbering, since the averaged images are counted from 1 again. The stacked
analysis names the reference images of every camera in `config_stacked.ini`, see below.

### Stacked multi-camera analysis

Several cameras that watch the same LED arrays can be inverted jointly, so that the rays
of all of them constrain one set of extinction coefficients:

```bash
python -m ledsa -conf_s   # writes config_stacked.ini
python -m ledsa -as
```

`config_stacked.ini` holds the solver and domain settings in `[DEFAULT]`, one
`[simulation_N]` section per camera, and a `[led_arrays]` section that maps a virtual
array ID to the array IDs of the individual runs.

Image IDs are counted within each run, so the same ID is a different point in time in
each camera. `num_ref_images` and `ref_img_indices` therefore override `[DEFAULT]` when
set in a `[simulation_N]` section, and `ref_img_indices = None` switches that camera back
to `num_ref_images`:

```ini
[simulation_1]
path            = /data/exp01/cam1
camera_position = -1.0 0.0 1.5
ref_img_indices = 385 386 387 388 389 390 391 392 393 394
```

Pick the reference images of all cameras from the same time window before the fire,
otherwise that camera gets an I0 of its own and biases its share of the inversion. The
header of the result file names the images that went into I0 per simulation.

### Corrupted images

Images that cannot be read - missing, empty, truncated or stored in an unsupported
format - are skipped instead of aborting the run. Every skipped image is reported on the
console and listed in `analysis/corrupted_images.csv` with the step that skipped it and
the reason:

```
image_path,analysis_step,error_type,error_message
imgs/IMG_0042.JPG,reading of the image metadata,Exiv2Error,Failed to read input data
imgs/IMG_0137.JPG,extraction of the LED intensities,OSError,image file is truncated
```

An image skipped while the metadata is read gets no entry in `image_infos.csv` and
`image_infos_analysis.csv`, so the analysis IDs stay a gapless sequence, as with
`num_skip_imgs`. An image skipped while the LED intensities are extracted keeps its entry
but drops out of the extinction coefficients, leaving a gap in the time axis rather than a
shift; the stacked analysis (`-as`) drops that time unless another image of the camera
lies within `time_sync_tolerance`. A restart with `-re` retries the skipped images.

`ref_img_id`, `first_img_experiment_id` and the time reference image stop the run with a
message naming the file instead of being skipped.

### Vertical images

A camera held on its side writes the pixels in the orientation of its sensor and notes how it
was held in the metadata, so a vertical photo is stored as a horizontal image. LEDSA turns every
image upright while it reads it, taking the rotation from the EXIF orientation of JPG and PNG
files and from the maker notes of the camera for RAW files. The LED positions, the search areas,
the plots and the coordinates computed from them are therefore all in the orientation the photo
was taken in.

Images that were exported without their metadata carry no orientation. `img_rotation` in
`config.ini` turns those, and overrides a wrong orientation:

| `img_rotation` | Rotation of the stored image |
|---|---|
| `auto` (default) | as the metadata of each image names it |
| `0` | none |
| `90`, `180`, `270` | that many degrees clockwise |

The rotation is part of reading an image, so step 1 and step 3 have to use the same one: changing
it means running the workflow again from `-s1`, because the search areas of the LEDs no longer fit
an image read in another orientation.

### Background correction (optional)

By default the extinction coefficients are computed from `sum_col_val`, the plain sum of
the pixel values in the search area of each LED. Step 3 additionally writes
background-corrected sums, which `reference_property` in `config_analysis.ini` selects, one
or several of them per run:

| `reference_property` | Background estimate |
|---|---|
| `sum_col_val` (default) | none |
| `bgsub_sum_col_val` | median of the pixels > 0 in a 2 px border ring of the search area, subtracted pixel-wise |
| `bgsub2_sum_col_val` | mean over the channel pixels (including 0) of the same border ring, subtracted as N·b |
| `sidebg_sum_col_val` | mean over the channel pixels in side bands next to the LED strip, subtracted as N·B |

N is the number of channel pixels in the search area. For small search areas the border
ring lies in the halo of the LED itself, whereas the side bands sample the scene next to
the strip.

The side bands run parallel to the strip, 25 to 60 px across it (`sideband_distances` in
`config.ini`), with the direction estimated for each LED from its nearest neighbours in
the same LED array; `sidebg_pedestal` holds the subtracted N·B. Pixels within
`search_area_radius` + `sideband_exclusion_margin` of another search area are excluded
from the bands, the margin defaulting to the search area radius - raise it above the
visible halo if the light of a neighbouring strip still reaches the bands.

`bgsub2_sum_col_val` and `sidebg_sum_col_val` are kept at least at `background_floor`
(default 5 %) of `sum_col_val`; values at this limit carry no information on the
extinction. `sidebg_sum_col_val` is NaN if the strip direction cannot be estimated (an LED
array with a single LED) or if the side bands contain no pixel, and the analysis stops
with an error naming these LEDs if the selected `reference_property` contains NaN.

### Temperature correction (optional)

An LED loses radiant power as it heats up, so the LEDs in the hot gas layer dim over the
course of an experiment even where no smoke is between them and the camera. Left
uncorrected, that drift enters the relative intensities as an extinction that is not there.

The factors the intensities are reduced by are determined outside LEDSA and handed to the
analysis as one file per camera channel, named in `config_analysis.ini`:

```ini
[temperature_correction]
   reduction_file_channel_0 = reduction_factors_red.csv
   reduction_file_channel_1 = None
   reduction_file_channel_2 = None
```

A channel with a file is corrected with it, a channel left at `None` is analysed as its
intensities were extracted. A `config_analysis.ini` written before the correction existed has
no such section and corrects nothing, so an older analysis is reproduced unchanged.

The file is read in either of two layouts. As a table, the first column holds the time in
seconds, the header of the other columns the height in metres, and every cell the factor the
intensity of that time and height is reduced by:

```
Time / s,3.3,3.2,3.1,2.9,2.6
0,1.0,1.0,1.0,1.0,1.0
60,0.991,0.993,0.994,0.996,0.998
```

As a list, the file has the three columns `height`, `time` and `reduction_factor` in any
order and one row per pair of a height and a time, which LEDSA pivots into the same table:

```
height,time,reduction_factor
0.0,0.0,1.0
0.0,100.0,0.85
1.0,0.0,1.0
1.0,100.0,0.85
```

The time is the experiment time in seconds, counted from `start_time` of `config.ini`. That is the
same reference as `Experiment_Time[s]` of `analysis/image_infos_analysis.csv`, the times the factors
are interpolated to, so a file written against the clock has to be shifted by `start_time` before it
is handed to LEDSA.

Neither layout has to be sorted, so the heights may be written in the order of the LEDs of
the array. LEDSA interpolates the table bilinearly to the experiment time of every image and the
height of every LED and divides the intensities by the result. The correction runs before the
reference intensities are averaged, so that I0 is a corrected intensity as well and the
factors of the reference images cancel out of the relative intensities.

The interpolation stays between the tabulated times and heights. An image or an LED outside them
keeps the intensity it was extracted with, because the table states where the reduction of the LEDs
was determined and there is nothing outside it to extrapolate from. A warning names the range the
table covers and how many values lie outside it, so a table that spans only part of the LED array
or only the first minutes of the experiment corrects only that part of the data.

The header of every result file states whether its channel was corrected and with which
factors:

```
# Reference_property: sum_col_val, ..., Temperature_correction: channel 0 from reduction_factors_red.csv (0.0 s to 1200.0 s, 2.60 m to 3.30 m), LEDSA 0.11.0
```

The name of the result file does not change, so a corrected run replaces the uncorrected
result of the same channel, reference property and solver, as a changed solver parameter does.

The factors belong to the LEDs rather than to a camera, so `config_stacked.ini` takes the
same section and every camera of a stacked inversion is corrected with the file of its
channel.

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
