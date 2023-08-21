class LEDAnalysisData:
    """
    Represents LED analysis data, including its physical properties and fit results.

    :ivar led_id: The ID of the LED.
    :vartype led_id: int
    :ivar led_array: The array index the LED belongs to.
    :vartype led_array: int
    :ivar fit_leds: Indicates whether to fit LEDs or not.
    :vartype fit_leds: bool
    :ivar led_center_x: X-coordinate of LED's center.
    :vartype led_center_x: float
    :ivar led_center_y: Y-coordinate of LED's center.
    :vartype led_center_y: float
    :ivar mean_color_value: Mean color value of the LED over the search area.
    :vartype mean_color_value: float
    :ivar sum_color_value: Integrated color value of the LED over the search area.
    :vartype sum_color_value: float
    :ivar max_color_value: Maximum color value observed for the LED.
    :vartype max_color_value: float
    :ivar fit_results: Fit results after fitting.
    :vartype fit_results: OptimizeResult
    :ivar fit_time: Time taken for fitting.
    :vartype fit_time: float
    """
    def __init__(self, led_id, led_array, fit_leds):
        """
        Initializes the LEDAnalysisData instance.

        :param led_id: ID of the LED.
        :type led_id: int
        :param led_array: Array index the LED belongs to.
        :type led_array: int
        :param fit_leds: Indicates whether to fit LEDs or not.
        :type fit_leds: bool
        """
        self.led_id = led_id
        self.led_array = led_array
        self.led_center_x = None
        self.led_center_y = None
        self.mean_color_value = None
        self.sum_color_value = None
        self.max_color_value = None
        self.fit_leds = fit_leds
        self.fit_results = None
        self.fit_time = None

    def __str__(self) -> str:
        """
        Generates a string representation of the LED analysis data with optional added output parameters of fitting.

        :return: String representation of all LED data.
        :rtype: str
        """
        out_str = self.get_main_data_string()
        if self.fit_leds:
            out_str += self.get_fit_data_string()
        out_str += "\n"
        return out_str

    def get_main_data_string(self) -> str:
        """
        Provides the main LED data as a formatted string.

        :return: Formatted string containing led_id, led_array and pixel values.
        :rtype: str
        """
        out_str = f'{self.led_id:4d},{self.led_array:2d},'
        out_str += f'{self.sum_color_value:10.4e},{self.mean_color_value:10.4e},{self.max_color_value}'
        return out_str

    def get_fit_data_string(self) -> str:
        """
        Provides the fitted LED data as a formatted string.

        :return: Formatted string of fitted LED data.
        :rtype: str
        """
        x, y, dx, dy, A, alpha, wx, wy = self.fit_results.x

        out_str = f'{self.led_center_x:10.4e}, {self.led_center_y:10.4e},'
        out_str += f'{x:10.4e},{y:10.4e},{dx:10.4e},{dy:10.4e},{A:10.4e},'
        out_str += f'{alpha:10.4e},{wx:10.4e},{wy:10.4e},{self.fit_results.success:12d},{self.fit_results.fun:10.4e},'
        out_str += f'{self.fit_results.nfev:9d},{self.fit_time:10.4e}'
        return out_str
