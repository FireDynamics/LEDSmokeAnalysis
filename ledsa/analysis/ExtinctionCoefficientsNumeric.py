import numpy as np
from scipy.optimize import minimize

from ledsa.analysis.Experiment import Experiment, Layers, Camera
from ledsa.analysis.ExtinctionCoefficients import ExtinctionCoefficients


class ExtinctionCoefficientsNumeric(ExtinctionCoefficients):
    """
    ExtinctionCoefficientsNumeric class.

    :ivar bounds: Bounds for each layer.
    :vartype bounds: list[tuple]
    :ivar weighting_preference: Weighting factor for the preference to push the numerical
    :vartype weighting_preference: float
    :ivar weighting_preference: Weighting factor for the preference to push the numerical solver to high or low values for the extinction coeffiientes.
    :vartype weighting_curvature: float
    :ivar num_iterations: Maximum number of iterations of the numerical solver.
    :vartype num_iterations: int
    :ivar type: Type of method.
    :vartype type: str
    """
    def __init__(self, experiment=Experiment(layers=Layers(20, 1.0, 3.35), camera=Camera(pos_x=4.4, pos_y=2, pos_z=2.3),
                                             led_array=3, channel=0),
                 reference_property='sum_col_val', num_ref_imgs=10, average_images=False, weighting_curvature=1e-6,
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
        :param weighting_preference: Weighting factor for the preference to push the numerical solver to high or low values for the extinction coeffiientes.
        :type weighting_preference: float
        :param num_iterations: Maximum number of iterations of the numerical solver.
        :type num_iterations: int
        """

        super().__init__(experiment, reference_property, num_ref_imgs, average_images)
        self.bounds = [(0, 10) for _ in range(self.experiment.layers.amount)]
        self.weighting_preference = weighting_preference
        self.weighting_curvature = weighting_curvature
        self.num_iterations = num_iterations
        self.type = 'numeric'

    def calc_coefficients_of_img(self, rel_intensities: np.ndarray) -> np.ndarray:
        """
        Calculate the extinction coefficients for a single image based on a minimization procedure.

        :param rel_intensities: Array of relative change in intensity of every LED.
        :return: Array of the computed extinction coefficients
        :rtype: np.ndarray
        """
        if len(self.coefficients_per_image_and_layer) == 0:
            kappa0 = np.zeros(self.experiment.layers.amount)
        else:
            kappa0 = self.coefficients_per_image_and_layer[-1]
        fit = minimize(self.cost_function, kappa0, args=rel_intensities,
                       method='TNC', bounds=tuple(self.bounds),
                       options={'maxfun': self.num_iterations, 'gtol': 1e-5, 'disp': False})
        print(fit.message)
        kappas = np.flip(fit.x)
        return kappas

    def calc_intensities(self, kappas: np.ndarray) -> np.ndarray:
        """
        Calculate the intensities from a given set of extinction coefficients.
        Is called in the minimization of the cost function.

        :param kappas: An array of extinction coefficients.
        :type kappas: np.ndarray
        :return: An array of the calculated intensities.
        :rtype: np.ndarray
        """
        n_leds = self.experiment.led_number
        intensities = np.zeros(n_leds)
        for led in range(n_leds):
            intensity = 1.0
            for layer in range(len(self.distances_per_led_and_layer[led, :])):
                intensity = intensity * np.exp(-kappas[layer] * self.distances_per_led_and_layer[led, layer])
            intensities[led] = intensity
        return intensities

    def cost_function(self, kappas: np.ndarray, target: np.ndarray) -> float:
        """
        Calculate the cost based on the difference between the computed intensities and target intensities.
        The cost function aims to minimize the root mean square error (rmse) between the computed and target intensities,
        while also considering the smoothness of the solution (curvature) and boundaries of the coefficients (preference).

        :param kappas: Extinction coefficients.
        :type kappas: np.ndarray
        :param target: Target intensities.
        :type target: np.ndarray
        :return: Computed cost.
        :rtype: float
        """
        intensities = self.calc_intensities(kappas)
        rmse = np.sqrt(np.sum((intensities - target) ** 2)) / len(intensities)
        curvature = np.sum(np.abs(kappas[0:-2] - 2 * kappas[1:-1] + kappas[2:])) * len(
            intensities) * 2 * self.weighting_curvature  # TODO: Factor 2 in weighting factor?
        preference = np.sum(kappas) / len(kappas) * self.weighting_preference
        return rmse + curvature + preference
