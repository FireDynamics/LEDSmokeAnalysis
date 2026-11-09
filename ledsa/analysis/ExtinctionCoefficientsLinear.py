import numpy as np
from scipy.optimize import nnls

from ledsa.analysis.Experiment import Experiment, Layers, Camera
from ledsa.analysis.ExtinctionCoefficients import ExtinctionCoefficients

class ExtinctionCoefficientsLinear(ExtinctionCoefficients):
    """
    ExtinctionCoefficientsLinear computes the extinction coefficients by
    linearizing the Beer–Lambert law. Specifically, using the transformation:

        -ln(I_e / I_0) = sum_i (sigma_i * Delta s_{i})

    the problem is recast as a linear least squares system that can be solved
    efficiently. A Tikhonov regularization is added via a finite-difference operator
    to enforce smoothness in the solution.

    :ivar solver: Type of solver (linear or nonlinear).
    :vartype solver: str
    """

    def __init__(self, experiment, reference_property='sum_col_val', num_ref_imgs=10, average_images=False, ref_img_indices=None, lambda_reg=1e-3, n_summarize=2, temperature_correction=None):
        """
        Initialize the ExtinctionCoefficientsLinear object.

        :param experiment: Object representing the experimental setup.
        :type experiment: Experiment
        :param reference_property: Reference property to be analyzed.
        :type reference_property: str
        :param num_ref_imgs: Number of reference images.
        :type num_ref_imgs: int
        :param ref_img_indices: Indices of reference images to use. If None, use num_ref_imgs.
        :type ref_img_indices: list[int] or None
        :param average_images: Flag to determine if intensities are computed as an average from consecutive images.
        :type average_images: bool
        :param n_summarize: Number of consecutive images summarized into one when averaging.
        :type n_summarize: int
        :param lambda_reg: Regularization parameter for Tikhonov regularization.
        :type lambda_reg: float
        :param temperature_correction: Reduction factors of the LED intensities of the channel. None
            leaves the intensities as they were extracted.
        :type temperature_correction: TemperatureCorrection or None
        """
        super().__init__(experiment, reference_property, num_ref_imgs, ref_img_indices, average_images,
                         n_summarize, temperature_correction)
        self.lambda_reg = lambda_reg
        self.solver = 'linear'

    def calc_coefficients_of_img(self, rel_intensities: np.ndarray) -> np.ndarray:
        """
        Calculate the extinction coefficients for a single image using a linearized approach.
        This method solves the linear system derived from the Beer-Lambert law using
        non-negative least squares with Tikhonov regularization.

        LEDs whose relative intensity is at or below zero, or not finite, are left out: the intensities
        are not clipped at zero, so the signal of such an LED has vanished in the noise of the sensor,
        and it carries no information on the extinction. If no LED is left, the image yields NaN.

        Known limitation: where the transmission approaches the read noise of the sensor, the LEDs that
        are left are those whose noise happened to be positive, so the extinction coefficients come out
        too low. On the Warsaw tests V008 and V009 this was 14 to 31 % at 1.24 and 1.64 1/m, where the
        transmission of the red LEDs was 0.1 to 0.2 % of I0.

        :param rel_intensities: Array of relative (normalized) LED intensities (I_e/I_0).
        :type rel_intensities: np.ndarray
        :return: Array of the computed extinction coefficients (sigmas).
        :rtype: np.ndarray
        """
        # Avoid log(0) by clipping intensities to a small positive value
        eps = 1e-10
        rel_intensities_clipped = np.clip(rel_intensities, eps, 1.0)

        # Compute the optical depth vector: b = -ln(I_e/I_0)
        # This linearizes the Beer-Lambert law: I_e/I_0 = exp(-sum(sigma_i * distance_i))
        optical_depths = -np.log(rel_intensities_clipped)

        # Get dimensions of the problem
        n_leds, n_layers = self.distances_per_led_and_layer.shape

        # A relative intensity at or below zero is not a measurement: the LED has vanished in the noise,
        # and the clamp above turns it into an optical depth of about 23, which would dominate the solve.
        # Such an LED is discarded by a weight of 0, which zeroes its row of the system.
        measured = np.isfinite(rel_intensities) & (np.asarray(rel_intensities) > 0.0)
        if not measured.any():
            return np.full(n_layers, np.nan)
        weights = measured.astype(float)

        # Build finite-difference matrix L for second derivative (curvature penalty)
        # This matrix approximates the second derivative and is used for regularization
        # to enforce smoothness in the solution
        if n_layers >= 3:
            L = np.zeros((n_layers - 2, n_layers))
            for i in range(n_layers - 2):
                # Coefficients for second-order finite difference approximation
                L[i, i] = 1
                L[i, i + 1] = -2
                L[i, i + 2] = 1
        else:
            # Fallback: use an identity if there aren't enough layers
            L = np.eye(n_layers)

        sqrt_lambda = np.sqrt(self.lambda_reg)

        # Augment the system with regularization
        # A_aug = [A; sqrt(lambda)*L], where A is the distance matrix
        A_aug = np.vstack((self.distances_per_led_and_layer * weights[:, None], sqrt_lambda * L))

        # Augment the right-hand side: b_aug = [b; 0]. The optical depth of a left out LED is set to 0 rather
        # than multiplied by its weight: that of an LED without a finite value is NaN, and NaN * 0 is NaN.
        b_aug = np.hstack((np.where(measured, optical_depths, 0.0), np.zeros(L.shape[0])))

        # Solve the non-negative least squares problem: min ||A_aug*x - b_aug||^2 subject to x >= 0
        sigmas, residuals = nnls(A_aug, b_aug)
        # sigmas, residuals, rank, s = np.linalg.lstsq(A_aug, b_aug, rcond=None)

        return sigmas

    # For investigation
    # import os

    # def calc_coefficients_of_img(self, rel_intensities: np.ndarray) -> np.ndarray:
    #     """
    #     Calculate the extinction coefficients for a single image using a linearized approach.
    #     This method solves the linear system derived from the Beer-Lambert law using
    #     non-negative least squares with Tikhonov regularization.
    #
    #     :param rel_intensities: Array of relative (normalized) LED intensities (I_e/I_0).
    #     :type rel_intensities: np.ndarray
    #     :return: Array of the computed extinction coefficients (sigmas).
    #     :rtype: np.ndarray
    #     """
    #     # Avoid log(0) by clipping intensities to a small positive value
    #     eps = 1e-10
    #     target = np.clip(rel_intensities, eps, 1.0)
    #
    #     # Compute the optical depth vector: b = -ln(I_e/I_0)
    #     # This linearizes the Beer-Lambert law: I_e/I_0 = exp(-sum(sigma_i * distance_i))
    #     optical_depths = -np.log(target)
    #
    #     # Get dimensions of the problem
    #     n_leds, n_layers = self.distances_per_led_and_layer.shape
    #
    #     # Build finite-difference matrix L for second derivative (curvature penalty)
    #     # This matrix approximates the second derivative and is used for regularization
    #     # to enforce smoothness in the solution
    #     if n_layers >= 3:
    #         L = np.zeros((n_layers - 2, n_layers))
    #         for i in range(n_layers - 2):
    #             # Coefficients for second-order finite difference approximation
    #             L[i, i] = 1
    #             L[i, i + 1] = -2
    #             L[i, i + 2] = 1
    #     else:
    #         # Fallback: use an identity if there aren't enough layers
    #         L = np.eye(n_layers)
    #
    #     # Interate lambda_reg from 1e-4 to 1
    #     logspace = np.logspace(-4, 0, 50)
    #
    #     os.chdir('/Users/kristian/Documents/NextVis/25_ledsa/lambda_iteration')
    #     for lambda_reg in logspace:
    #         sqrt_lambda = np.sqrt(lambda_reg)
    #
    #         # Augment the system with regularization
    #         # A_aug = [A; sqrt(lambda)*L], where A is the distance matrix
    #         A_aug = np.vstack((self.distances_per_led_and_layer, sqrt_lambda * L))
    #
    #         # Augment the right-hand side: b_aug = [b; 0]
    #         b_aug = np.hstack((optical_depths, np.zeros(L.shape[0])))
    #
    #         # Solve the non-negative least squares problem: min ||A_aug*x - b_aug||^2 subject to x >= 0
    #         sigmas, residuals = nnls(A_aug, b_aug)
    #         np.savetxt(f'lambda_{lambda_reg:.4f}.txt', sigmas)
    #     return sigmas