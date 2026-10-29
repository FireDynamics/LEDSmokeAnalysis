import warnings

import numpy as np
from scipy.optimize import minimize

from ledsa.analysis.Experiment import Experiment, Layers, Camera
from ledsa.analysis.ExtinctionCoefficients import ExtinctionCoefficients

# Exit status of scipy's TNC implementation for an exhausted evaluation budget
MAX_FUN_REACHED = 3


class ExtinctionCoefficientsNonLinear(ExtinctionCoefficients):
    """
    ExtinctionCoefficientsNonLinear computes the extinction coefficients by
    directly minimizing the difference between observed and predicted intensities
    using the Beer-Lambert law:

        I_e = I_0 * exp(-sum_i (sigma_i * Delta s_{i}))

    A numerical optimization approach is used with regularization terms to enforce
    smoothness in the solution. The cost function is

        rmse(sigma) + weighting_curvature * mean((D2 sigma)^2)
                    + weighting_preference * mean(sigma),

    where rmse is taken over the LEDs and D2 is the second difference over the layers. Every
    term is a mean, so the weights keep their meaning for any number of LEDs and layers.

    :ivar bounds: Bounds for each layer.
    :vartype bounds: list[tuple]
    :ivar weighting_preference: Weighting factor for the preference to push the numerical solver to high or low values for the extinction coefficients.
    :vartype weighting_preference: float
    :ivar weighting_curvature: Weighting factor for the smoothness of the solution.
    :vartype weighting_curvature: float
    :ivar num_iterations: Maximum number of iterations of the numerical solver.
    :vartype num_iterations: int
    :ivar curvature_operator: Finite difference operator of the second derivative over the layers.
    :vartype curvature_operator: np.ndarray
    :ivar solver: Type of solver (linear or nonlinear).
    :vartype type: str
    """
    def __init__(self, experiment, reference_property='sum_col_val', num_ref_imgs=10, ref_img_indices=None, average_images=False, weighting_curvature=1e-2,
                 weighting_preference=-6e-3, num_iterations=1000, n_summarize=2,
                 temperature_correction=None):
        """
        :param experiment: Object representing the experimental setup.
        :type experiment: Experiment
        :param reference_property: Reference property to be analysed
        :type reference_property: str
        :param num_ref_imgs: Number of reference images.
        :type num_ref_imgs: int
        :param ref_img_indices: Indices of reference images to use. If None, use num_ref_imgs.
        :type ref_img_indices: list[int] or None
        :param average_images: Flag to determine if intensities are computed as an average from two consecutive images.
        :type average_images: bool
        :param weighting_curvature: Weighting factor for the smoothness of the solution.
        :type weighting_curvature: float
        :param weighting_preference: Weighting factor for the preference to push the numerical solver to high or low values for the extinction coefficients.
        :type weighting_preference: float
        :param num_iterations: Maximum number of iterations of the numerical solver.
        :type num_iterations: int
        :param n_summarize: Number of consecutive images summarized into one when averaging.
        :type n_summarize: int
        :param temperature_correction: Reduction factors of the LED intensities of the channel. None
            leaves the intensities as they were extracted.
        :type temperature_correction: TemperatureCorrection or None
        """

        # the nonlinear solver has no weighting, so it is switched off whatever the default is
        super().__init__(experiment, reference_property, num_ref_imgs, ref_img_indices, average_images,
                         n_summarize, temperature_correction, snr_full_weight=None)
        self.bounds = [(0, 10) for _ in range(self.experiment.layers.amount)]
        self.weighting_preference = weighting_preference
        self.weighting_curvature = weighting_curvature
        self.num_iterations = num_iterations
        self.curvature_operator = second_difference_operator(self.experiment.layers.amount)
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
        # to find the extinction coefficients that minimize the cost function.
        # The gradient is handed over explicitly: TNC falls back to finite differences otherwise,
        # and those are too inaccurate here to reach the minimum, see cost_gradient.
        fit = minimize(self.cost_function, sigma0, args=rel_intensities, jac=self.cost_gradient,
                       method='TNC', bounds=tuple(self.bounds),
                       options={'maxfun': self.num_iterations, 'gtol': 1e-5, 'disp': False})
        # A used up budget is the one exit that costs accuracy and that the user can do something
        # about. The other exits of TNC report a minimum it cannot improve on, which is where the
        # solution of an undisturbed image lies: its intensities match their target exactly, so the
        # rmse term sits at the kink of its square root, where no direction leads downhill.
        if fit.status == MAX_FUN_REACHED:
            warnings.warn(f'The minimization of the cost function stopped after {fit.nfev} evaluations: '
                          f'{fit.message}. The extinction coefficients of this image are the ones the '
                          f'solver reached last, not the minimum. Raise num_iterations in '
                          f'config_analysis.ini.', category=UserWarning, stacklevel=2)
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
        # I = I_0 * exp(-sum_i(sigma_i * distance_i)) over the layers every LED is seen through
        return np.exp(-self.distances_per_led_and_layer @ sigmas)

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
        # Root mean square error between the intensities of the current sigmas and the target ones
        rmse = np.sqrt(np.mean((self.calc_intensities(sigmas) - target) ** 2))

        # Curvature penalty on the squared second differences, which enforces smoothness.
        # Squares keep the term differentiable: with the absolute values used before, the minimizer
        # stalled on the kinks the penalty itself drives the solution onto.
        second_differences = self.curvature_operator @ sigmas
        if second_differences.size == 0:
            curvature = 0.0
        else:
            curvature = self.weighting_curvature * np.mean(second_differences ** 2)

        # Preference term to bias the solution toward higher or lower values
        preference = self.weighting_preference * np.mean(sigmas)

        # Total cost is the sum of all terms
        return rmse + curvature + preference

    def cost_gradient(self, sigmas: np.ndarray, target: np.ndarray) -> np.ndarray:
        """
        Calculate the gradient of the cost function with respect to the extinction coefficients.

        The minimizer needs the gradient of every term of :meth:`cost_function`. Left to estimate
        it from finite differences, it stops at points that are far from the minimum: the
        intensities fall off exponentially, so the difference quotients are dominated by rounding
        error, and the stopping criteria of TNC are met long before the solution is reached.

        :param sigmas: Extinction coefficients (sigma values).
        :type sigmas: np.ndarray
        :param target: Target relative intensities (I_e/I_0).
        :type target: np.ndarray
        :return: Gradient of the cost with respect to every extinction coefficient.
        :rtype: np.ndarray
        """
        intensities = self.calc_intensities(sigmas)
        residuals = intensities - target
        rmse = np.sqrt(np.mean(residuals ** 2))
        if rmse == 0:
            # The intensities match their target, which is the minimum of the rmse term
            rmse_gradient = np.zeros(len(sigmas))
        else:
            # d(rmse)/d(sigma_i) with d(intensity)/d(sigma_i) = -distance_i * intensity
            rmse_gradient = -(self.distances_per_led_and_layer.T @ (residuals * intensities)) / (
                    len(residuals) * rmse)

        second_differences = self.curvature_operator @ sigmas
        if second_differences.size == 0:
            # Fewer than three layers have no second difference and so no curvature
            curvature_gradient = np.zeros(len(sigmas))
        else:
            curvature_gradient = (2 * self.weighting_curvature / len(second_differences) *
                                  (self.curvature_operator.T @ second_differences))

        preference_gradient = self.weighting_preference / len(sigmas)

        return rmse_gradient + curvature_gradient + preference_gradient


def second_difference_operator(num_layers: int) -> np.ndarray:
    """
    Build the finite difference operator of the second derivative over the layers.

    :param num_layers: Number of layers.
    :type num_layers: int
    :return: Matrix that maps the extinction coefficients to their second differences. It has no
        rows if fewer than three layers are given, which have no second difference.
    :rtype: np.ndarray
    """
    num_second_differences = max(num_layers - 2, 0)
    operator = np.zeros((num_second_differences, num_layers))
    for row in range(num_second_differences):
        operator[row, row:row + 3] = (1, -2, 1)
    return operator
