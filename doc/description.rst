
Description
============

LEDSA (LEDSmokeAnalysis) is a python based software package for the computation of spatially and temporally resolved light extinction coefficients from photometric measurements. The method relies on capturing the change in intensity of individual light sources due to fire induced smoke. Images can be accquired within laboratory experiments using commercially available digital cameras. Both conventional JPEG images and raw image data can be used for the procedure. However, the latter are required to allow a reliable photometric measurement.

Based on the Beer-Lambert law a model for the light transmission is formulated as a line of sight integration of the extinction coefficients between the camera and the individual LEDs. Accordingly, the region of interest is discretized by horizontal layers, each characterized by a single extinction coefficient. The underlying assumption of the model is that the smoke properties are homogeneous within these layers. Based on an inverse modelling approach local values of the extinction coefficient can be deduced by fitting the model intensities to the measured LED intensities.

Acquiring continuous series of images of the LED setup allows the light extinction coefficients to be determined on a temporal scale. The subsequent data analysis can be performed independently for the different color channels of the camera. This allows the smoke characteristics to be studied as a function the wavelength of light.

The user must supply the geometric coordinates of the camera and the light sources. Furthermore, the boundary conditions of the model, such as the number and size of the layers, must be defined.
