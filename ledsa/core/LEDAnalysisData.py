class LEDAnalysisData:
    """Data class for step 3 for one led and image. Includes string conversion."""

    def __init__(self, led_id: int, led_array: int, fit_leds: bool):
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

    def __str__(self):
        out_str = self.get_main_data_string()
        if self.fit_leds:
            out_str += self.get_fit_data_string()
        out_str += "\n"
        return out_str

    def get_main_data_string(self):
        out_str = f'{self.led_id:4d},{self.led_array:2d},'
        out_str += f'{self.sum_color_value:10.4e},{self.mean_color_value:10.4e},{self.max_color_value}'
        return out_str

    def get_fit_data_string(self):
        x, y, dx, dy, A, alpha, wx, wy = self.fit_results.x

        out_str = f'{self.led_center_x:10.4e}, {self.led_center_y:10.4e},'
        out_str += f'{x:10.4e},{y:10.4e},{dx:10.4e},{dy:10.4e},{A:10.4e},'
        out_str += f'{alpha:10.4e},{wx:10.4e},{wy:10.4e},{self.fit_results.success:12d},{self.fit_results.fun:10.4e},'
        out_str += f'{self.fit_results.nfev:9d},{self.fit_time:10.4e}'
        return out_str
