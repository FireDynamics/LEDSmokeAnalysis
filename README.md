# LEDSmokeAnalysis

How to use LEDSA:

Make ledsa.py available on your operating system. The easiest way is to copy it into your working directory.

Set your PYTHONPATH to find led_helper.py, ledsa.py and ledsa_conf.py so they can be imported

If you don't have a configuration file already in your working directory, run python ledsa.py -config

Open config.ini and add every information already available (there is no fallback for every unset variable implemented yet and it will be faster, doing it this way)
If you don't know, what a variable means or is used for or aren't sure of the formatting, don't change its value, there is probably a fallback for it.
    
It is a good idea to make a template configuration file if multiple, similar experiments are analysed.

Run python ledsa.py -s1

Run python ledsa.py -s2. You will be asked to write the different indices of the edges of the led arrays into the shell.
You can do it also after step1 directly in the config file. The information is found in ./plots/led_search_areas.plot.pdf
    
Run python ledsa.py -s3. This will take quite a while, even on many cores. Grab something to eat.





TODO:

whole script with UI for all steps and parameters

change path operations to pathlib

LOGIC:

Hough transformation -> get lines

find xyz-space (from plane) (median of all cameras? seems complicated if leds are not labeled the same[missing led from one angle])

