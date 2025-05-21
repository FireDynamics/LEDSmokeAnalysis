import numpy as np
from scipy.optimize import minimize

from ledsa.analysis.Experiment import Experiment, Layers, Camera
from ledsa.analysis.ExtinctionCoefficients import ExtinctionCoefficients


class ExtinctionCoefficientsNonLinear(ExtinctionCoefficients):
    """
    ExtinctionCoefficientsNonLinear computes the extinction coefficients by
    directly minimizing the difference between observed and predicted intensities
    using the Beer-Lambert law:

        I_e = I_0 * exp(-sum_i (sigma_i * Delta s_{i}))

    A numerical optimization approach is used with regularization terms to enforce
    smoothness in the solution.

    :ivar bounds: Bounds for each layer.
    :vartype bounds: list[tuple]
    :ivar weighting_preference: Weighting factor for the preference to push the numerical solver to high or low values for the extinction coefficients.
    :vartype weighting_preference: float
    :ivar weighting_curvature: Weighting factor for the smoothness of the solution.
    :vartype weighting_curvature: float
    :ivar num_iterations: Maximum number of iterations of the numerical solver.
    :vartype num_iterations: int
    :ivar solver: Type of solver (linear or nonlinear).
    :vartype type: str
    """
    def __init__(self, experiment, reference_property='sum_col_val', num_ref_imgs=10, average_images=False, weighting_curvature=1e-6,
                 weighting_preference=-6e-3, num_iterations=200):
        """
        :param experiment: Object representing the experimental setup.
        :type experiment: Experiment
        :param reference_property: Reference property to be analysed
        :type reference_property: str
        :param num_ref_imgs: Number of reference images.
        :type num_ref_imgs: int
        :param average_images: Flag to determine if intensities are computed as an average from two consecutive images.
        :type average_images: bool
        :param weighting_curvature: Weighting factor for the smoothness of the solution.
        :type weighting_curvature: float
        :param weighting_preference: Weighting factor for the preference to push the numerical solver to high or low values for the extinction coefficients.
        :type weighting_preference: float
        :param num_iterations: Maximum number of iterations of the numerical solver.
        :type num_iterations: int
        """

        super().__init__(experiment, reference_property, num_ref_imgs, average_images)
        self.bounds = [(0, 10) for _ in range(self.experiment.layers.amount)]
        self.weighting_preference = weighting_preference
        self.weighting_curvature = weighting_curvature
        self.num_iterations = num_iterations
        self.solver = 'nonlinear'

    def calc_coefficients_of_img(self, rel_intensities: np.ndarray) -> np.ndarray:
        """
        Calculate the extinction coefficients for a single image based on a minimization procedure.

        :param rel_intensities: Array of relative (normalized) LED intensities (I_e/I_0).
        :return: Array of the computed extinction coefficients (sigmas)
        :rtype: np.ndarray
        """
        # Initialize starting point for optimization
        # If this is the first image, start with zeros
        # Otherwise, use coefficients from previous image as initial guess
        if len(self.coefficients_per_image_and_layer) == 0:
            sigma0 = np.zeros(self.experiment.layers.amount)
        else:
            sigma0 = self.coefficients_per_image_and_layer[-1]

        # Use TNC (Truncated Newton Conjugate) optimization method with bounds
        # to find the extinction coefficients that minimize the cost function
        fit = minimize(self.cost_function, sigma0, args=rel_intensities,
                       method='TNC', bounds=tuple(self.bounds),
                       options={'maxfun': self.num_iterations, 'gtol': 1e-5, 'disp': False})
        print(fit.message)
        sigmas = fit.x
        return sigmas

    def calc_intensities(self, sigmas: np.ndarray) -> np.ndarray:
        """
        Calculate the intensities from a given set of extinction coefficients.
        This implements the Beer-Lambert law and is called during the minimization of the cost function.

        :param sigmas: An array of extinction coefficients (sigma values).
        :type sigmas: np.ndarray
        :return: An array of the calculated relative intensities (I_e/I_0).
        :rtype: np.ndarray
        """
        # Get the number of LEDs on the LED array
        n_leds = self.experiment.num_leds
        intensities = np.zeros(n_leds)

        # Calculate intensity for each LED
        for led in range(n_leds):
            # Start with full intensity (I_0)
            intensity = 1.0

            # Apply Beer-Lambert law for each layer the light passes through
            for layer in range(len(self.distances_per_led_and_layer[led, :])):
                # I = I_0 * exp(-sigma * distance)
                intensity = intensity * np.exp(-sigmas[layer] * self.distances_per_led_and_layer[led, layer])

            intensities[led] = intensity
        return intensities

    def cost_function(self, sigmas: np.ndarray, target: np.ndarray) -> float:
        """
        Calculate the cost based on the difference between the computed intensities and target intensities.
        The cost function aims to minimize the root mean square error (rmse) between the computed and target intensities,
        while also considering the smoothness of the solution (curvature) and boundaries of the coefficients (preference).

        :param sigmas: Extinction coefficients (sigma values).
        :type sigmas: np.ndarray
        :param target: Target relative intensities (I_e/I_0).
        :type target: np.ndarray
        :return: Computed cost.
        :rtype: float
        """
        # Calculate predicted intensities using current sigma values
        intensities = self.calc_intensities(sigmas)

        # Calculate root mean square error between predicted and target intensities
        rmse = np.sqrt(np.sum((intensities - target) ** 2)) / len(intensities)

        # Calculate curvature penalty (second derivative) to enforce smoothness
        # This is a finite difference approximation of the second derivative
        curvature = np.sum(np.abs(sigmas[0:-2] - 2 * sigmas[1:-1] + sigmas[2:])) * len(
            intensities) * 2 * self.weighting_curvature

        # Add preference term to bias solution toward higher or lower values
        preference = np.sum(sigmas) / len(sigmas) * self.weighting_preference

        # Total cost is the sum of all terms
        return rmse + curvature + preference
