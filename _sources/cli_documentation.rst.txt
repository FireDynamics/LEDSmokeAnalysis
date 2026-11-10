CLI Documentation
=================

General Usage
-------------

To use the LEDSA CLI, the general structure is as follows::

   python -m ledsa [ARGUMENT] [OPTIONS]

Arguments
---------

Data Extraction
^^^^^^^^^^^^^^^

.. list-table::
   :widths: 20 50 30
   :header-rows: 1

   * - Argument
     - Description
     - Options
   * - ``-conf``, ``--config``
     - Create the data extraction configuration file.
     - --
   * - ``--s1``, ``--step_1``, ``--find_search_areas``
     - Step 1: Analyze a reference image to find the LEDs and create a labeled plot with the search areas.
     - --
   * - ``-s2``,  ``--step_2``,  ``--analyse_positions``
     - Step 2: Match the LEDs to the arrays according to the line edge indices in the config file.
     - --
   * - ``-s3``, ``--step_3`, ``--analyse_photo``
     - Step 3: Analyse all images to extract pixel values inside the search areas and fit them to an LED model (SLOW!).
     - ``-r`` (Use the red channel, default), ``-g`` (Use the green channel), ``-b`` (Use the blue channel), ``-rgb`` (Run for each channel)
   * - ``-s3_fast``, ``--step_3_fast`, ``--analyse_photo_fast``
     - Step 3: Analyse all images to extract pixel values and calculate the accumulated values inside the search areas (FAST!).
     - Channel flags as for ``-s3``
   * - ``-re``, ``--restart``
     - Restart step 3 if it was previously interrupted. Only the images that lack the result file of one of the selected channels are analysed, in all of these channels. The restart resumes in the mode of the interrupted run, with the LED fits after ``-s3`` and without them after ``-s3_fast``, as the headers of its result files tell. If there are no result files yet, it runs without the fits. If the result files were written in both modes, it stops, because it cannot tell which mode to resume.
     - The channel flags of the interrupted run, as for ``-s3``


Analysis
^^^^^^^^

.. list-table::
   :widths: 20 50 30
   :header-rows: 1

   * - Argument
     - Description
     - Options
   * - ``-conf_a``, ``--config_analysis``
     - Creates the analysis configuration file.
     - --
   * - ``-a``, ``--analysis``
     - Computes the extinction coefficients of every quantity ``reference_property`` names in ``config_analysis.ini``, each into ``analysis/extinction_coefficients/<solver>/<quantity>/``.
     - --
   * - ``--cc``
     - Applies the color correction matrix before calculating the extinction coefficients. Use only if the reference property is not already color corrected.
     - --
   * - ``--cc_channels``
     - Specifies the channels to which color correction gets applied.
     - Can extend the list of channels, Multiple values allowed.

.. warning::
    Color Correction is in a test state and has not yet been sufficiently evaluated. The application may lead to incorrect results.

Stacked Multi-Camera Analysis
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :widths: 20 50 30
   :header-rows: 1

   * - Argument
     - Description
     - Options
   * - ``-conf_s [FILENAME]``, ``--config_stacked [FILENAME]``
     - Create a template ``config_stacked.ini`` for multi-camera stacked analysis. The optional argument sets the output filename (default: ``config_stacked.ini``).
     - ``--n_simulations N`` sets the number of simulation blocks in the template (default: 2).
   * - ``--n_simulations N``
     - Number of ``[simulation_N]`` blocks written into the template ``config_stacked.ini``. Only used together with ``-conf_s``.
     - Default: 2
   * - ``-as``, ``--analysis_stacked``
     - Run the stacked multi-camera extinction coefficient calculation using ``config_stacked.ini`` in the current directory, for every quantity its ``reference_property`` names.
     - --

Coordinates
^^^^^^^^^^^

.. list-table::
   :widths: 20 50 30
   :header-rows: 1

   * - Argument
     - Description
     - Options
   * - ``-coord``, ``--coordinates``
     - Calculate the 3D coordinates of the LEDs based on the calculated pixel positions and the physical edge coordinates of the LED arrays.
     - --

Testing
^^^^^^^

.. list-table::
   :widths: 20 50 30
   :header-rows: 1

   * - Argument
     - Description
     - Options
   * - ``-atest``, ``--acceptance-test``
     - Run the acceptance test suite.
     - --
   * - ``-atest_debug``, ``--acceptance-test_debug``
     - Run the acceptance test suite in debug mode.
     - --

Demo
^^^^

.. list-table::
   :widths: 20 50 30
   :header-rows: 1

   * - Argument
     - Description
     - Options
   * - ``-d``, ``--demo``
     - Flag to indicate that the LEDSA demo should be run.
     - Must be used with ``--setup`` or ``--run``
   * - ``--setup``
     - Create the required directories for the simulations and download images and config files.
     - Optional: Path to setup simulation and image directories. Default to ``'.'``
   * - ``--run``
     - Run the LEDSA demo in the current working directory.
     - --