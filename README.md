# LEDSmokeAnalysis

LEDSmokeAnalysis (LEDSA) is scientific program to analyse the extinction of light due to smoke in the air from image data. An experimental setup is needed, in which a camera captures pictures of lines of leds. The software is designed to improve our understanding of the propagation of smoke and the visibiltiy in rooms filled with smoke during a fire. For further information about how the program works and the experimental setup see [papers]. An example for an experimental setup is shown below:\
[img setup]
The whole documentation can be found [link to doc]

## Installation

To install LEDSA python 3 and an acutal version of pip need to be installed.
Then run: 

`python3 -m pip install ledsa`

## Usage

LEDSA can be run directly from the console with

`python3 -m ledsa`.

### Configuration File

A configuration file is needed which defines the behaviour of LEDSA and some pathes etc. With the `--config` flag a default configuration file can be created inside the working directory. Further information about all the variables set in the config file can be found [here]. It is a good idea to make a template configuration file if multiple, similar experiments are analysed.

### Intensity Calculations

To find the intensity changes over multiple images, 3 steps are to be taken.\
Step 1 finds and labels each led from a reference image which shoud not contain any smoke. To do this run

`python3 -m ledsa --s1`.

Step 2 finds and labels the different led arrays. You will be asked to write the different indices of the edges of the led arrays into the shell, which will be saved in the config file. The information is found in ./plots/led_search_areas.plot.pdf, generated in step 1. The flag is `--s2`

Step 3 calculates a measure for the itensities for every image and every led. The Flag `--s3` fits a 2D Function over every led as described in [paper]. This is computational very expencive. `--s3_fast` counts the color values of each pixel for every led, which is much faster.

To calculate the extinction coefficients, the 3D coordinates of every led needs to be known. `--coord` calculates them from the coordinates of the edge leds used in step 2.

### Extinction Coefficient Calculations

To calculate the extinction coefficients run

`python3 -m ledsa.analysis`

after finishing step 3. Some setup data is needed like the position of the camera. `--default_input` creates a file for this information. 

## Contributing

To introduce new, tested, documented and stable changes, pull/ merge requests into the master branch are used.

Pull request drafts can be used to communicate about changes and new functionality.

After reviewing and testing the changes, they will be merged into master.

Every merge with master is followed by introducing a new version tag corresponding to the semantic versioning pradigm.
