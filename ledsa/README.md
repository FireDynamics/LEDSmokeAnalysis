# LEDSmokeAnalysis

<b>How to use LEDSA:</b>

Set your PYTHONPATH to find the ledsa folder so the package can be imported.
Now, the package can be accessed via import ledsa.
The easiest way to use ledsa is by executing the main script. To do it, use the -m flag for python to search for executable packages.

If you don't have a configuration file already in your working directory, run python ledsa.py -config

Open config.ini and add every information already available (there is no fallback for every unset variable implemented yet and it will be faster, doing it this way)
If you don't know, what a variable means or is used for or aren't sure of the formatting, don't change its value, there is probably a fallback for it.
    
It is a good idea to make a template configuration file if multiple, similar experiments are analysed.

Run python -m ledsa -s1 from anywhere

Run python -m ledsa -s2. You will be asked to write the different indices of the edges of the led arrays into the shell.
You can do it also after step1 directly in the config file. The information is found in ./plots/led_search_areas.plot.pdf
    
Run python -m ledsa -s3. This will take quite a while, even on many cores. Grab something to eat.

Run python -m ledsa -coord to calculate the 3D coordinates of the LEDs from the reference image. To be able to do
it,there must be the coordinates of the two outermost LEDs for each LED array saved in the variable
line_edge_coordinates inside config.ini


<b>How this git repo is structured:</b>

To introduce new, tested and stable changes, push requests to the development branch are used.

If the master branch is not ahead of the push and no other push request to master is open, the request can go directly to master instead.

Push request drafts to the development branch can be used to communicate about changes and new functionality.

After reviewing the changes, they will be merged into the branch.

Merges to development must be tested again after merging master into it before merging it to master. If there is an open push request to master, it must be merged with development and tested as well before pushing development to master.

Every merge with master is followed by introducing a new version tag corresponding to the semantic versioning pradigm.
