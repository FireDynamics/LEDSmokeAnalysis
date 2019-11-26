# LEDSmokeAnalysis

How to use LEDSA:

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





TODO:

whole script with UI for all steps and parameters

change path operations to pathlib

LOGIC:

Hough transformation -> get lines

find xyz-space (from plane) (median of all cameras? seems complicated if leds are not labeled the same[missing led from one angle])

write module to access raw data