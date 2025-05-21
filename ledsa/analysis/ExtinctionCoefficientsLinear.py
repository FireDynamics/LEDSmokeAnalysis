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

    def __init__(self, experiment, reference_property='sum_col_val', num_ref_imgs=10, average_images=False, lambda_reg=1e-3,):
        """
        Initialize the ExtinctionCoefficientsLinear object.

        :param experiment: Object representing the experimental setup.
        :type experiment: Experiment
        :param reference_property: Reference property to be analyzed.
        :type reference_property: str
        :param num_ref_imgs: Number of reference images.
        :type num_ref_imgs: int
        :param average_images: Flag to determine if intensities are computed as an average from consecutive images.
        :type average_images: bool
        :param lambda_reg: Regularization parameter for Tikhonov regularization.
        :type lambda_reg: float
        """
        super().__init__(experiment, reference_property, num_ref_imgs, average_images)
        self.lambda_reg = lambda_reg
        self.solver = 'linear'

    def calc_coefficients_of_img(self, rel_intensities: np.ndarray) -> np.ndarray:
        """
        Calculate the extinction coefficients for a single image using a linearized approach.
        This method solves the linear system derived from the Beer-Lambert law using
        non-negative least squares with Tikhonov regularization.

        :param rel_intensities: Array of relative (normalized) LED intensities (I_e/I_0).
        :type rel_intensities: np.ndarray
        :return: Array of the computed extinction coefficients (sigmas).
        :rtype: np.ndarray
        """
        # Avoid log(0) by clipping intensities to a small positive value
        eps = 1e-10
        target = np.clip(rel_intensities, eps, 1.0)

        # Compute the optical depth vector: b = -ln(I_e/I_0)
        # This linearizes the Beer-Lambert law: I_e/I_0 = exp(-sum(sigma_i * distance_i))
        optical_depths = -np.log(target)

        # Get dimensions of the problem
        n_leds, n_layers = self.distances_per_led_and_layer.shape

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
        A_aug = np.vstack((self.distances_per_led_and_layer, sqrt_lambda * L))

        # Augment the right-hand side: b_aug = [b; 0]
        b_aug = np.hstack((optical_depths, np.zeros(L.shape[0])))

        # Solve the non-negative least squares problem: min ||A_aug*x - b_aug||^2 subject to x >= 0
        sigmas, residuals = nnls(A_aug, b_aug)

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