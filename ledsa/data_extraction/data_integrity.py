import warnings

def check_intensity_normalization(ref_img_data, ref_intensities, reference_property, tolerance=0.05):
    rel_devs = (ref_img_data[reference_property] - ref_intensities).abs() / ref_intensities.abs()
    hits = rel_devs[rel_devs > tolerance]

    if hits.empty:
        return

    lines = [
        f"  img_id={img_id}, led_id={led_id}, rel. deviation = {val:.2%}"
        for (img_id, led_id), val in hits.items()
    ]

    msg = (
        f"In the process of normalisation {len(hits)} values exceed {tolerance:.0%} tolerance of relative deviation"
        f" against mean intensities! You might check the reference images.\n"
        + "\n".join(lines)
    )

    warnings.warn(msg, category=UserWarning, stacklevel=2)


def check_led_positions():
    pass

def check_led_saturations():
    pass
