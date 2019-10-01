# LEDSmokeAnalysis

TODO:

make LED class with id, coordinates (in pixel and xyz-space), array label 

module for configfile generation (module config parser)
Configfile:
	- eichtime
	- coordinates of line endpoints
	- 
	
module for folder generation (Windows/Linux)

install executable (Windows/Linux) (creating folder structure, install needed libs, moving all needed files)

whole script with UI for all steps and parameters

LOGIC:

Time from meta data -> compare with eichtime -> find experiment time

Hough transformation -> get lines

find xyz-space (from plane) (median of all cameras? seems complicated if leds are not labeled the same[missing led from one angle])

