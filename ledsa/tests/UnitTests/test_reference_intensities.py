"""Unit tests for the reference intensities, both from explicitly named images and from the first ones."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ledsa.analysis.ExtinctionCoefficients import ExtinctionCoefficients

REFERENCE_PROPERTY = 'sum_col_val'
LED_IDS = [0, 1]


class ExtinctionCoefficientsStub(ExtinctionCoefficients):
    """Smallest concrete solver, since only the reference intensities are of interest here."""

    def calc_coefficients_of_img(self, rel_intensities: np.ndarray) -> np.ndarray:
        return rel_intensities


class ExperimentStub:
    led_array = 0
    path = Path('.')


def make_solver(available_img_ids, ref_img_indices=None, num_ref_imgs=1,
                intensity_of_img=lambda img_id: 100.0):
    """Create a solver whose image data holds the given images only.

    :param available_img_ids: IDs of the images that hold intensity values.
    :param ref_img_indices: Reference image IDs requested by the configuration. None falls back to
        the first *num_ref_imgs* images.
    :param num_ref_imgs: Size of the reference window used when no indices are requested.
    :param intensity_of_img: Intensity of every LED of an image.
    """
    solver = ExtinctionCoefficientsStub(ExperimentStub(), reference_property=REFERENCE_PROPERTY,
                                       num_ref_imgs=num_ref_imgs, ref_img_indices=ref_img_indices)
    index = pd.MultiIndex.from_product([available_img_ids, LED_IDS], names=['img_id', 'led_id'])
    solver.calculated_img_data = pd.DataFrame(
        {REFERENCE_PROPERTY: [intensity_of_img(img_id) for img_id, _ in index]}, index=index)
    return solver


def test_available_reference_images_are_averaged_without_a_warning(recwarn):
    # The intensities of the reference images stay close to their mean, so that only a missing
    # reference image could warn here
    solver = make_solver([1, 2, 3], ref_img_indices=[2, 3],
                         intensity_of_img=lambda img_id: 100.0 + 2.0 * img_id)

    solver.calc_and_set_ref_intensities()

    assert list(solver.ref_intensities) == [105.0] * len(LED_IDS)
    assert [str(warning.message) for warning in recwarn] == []


def test_missing_reference_image_is_named_in_a_warning():
    solver = make_solver([1, 3], ref_img_indices=[2, 3])

    with pytest.warns(UserWarning, match=r'\[2\]'):
        solver.calc_and_set_ref_intensities()

    # The remaining image still gives reference intensities, so the analysis can go on
    assert list(solver.ref_intensities) == [100.0] * len(LED_IDS)


def test_reference_images_without_any_data_raise():
    solver = make_solver([1, 3], ref_img_indices=[4, 5])

    with pytest.raises(ValueError, match=r'\[4, 5\]'):
        solver.calc_and_set_ref_intensities()


class TestReferenceWindowOfTheFirstImages:
    """The reference window of num_ref_imgs is checked like explicitly named reference images."""

    def test_complete_window_is_averaged_without_a_warning(self, recwarn):
        solver = make_solver([1, 2, 3], num_ref_imgs=2,
                             intensity_of_img=lambda img_id: 100.0 + 2.0 * img_id)

        solver.calc_and_set_ref_intensities()

        assert list(solver.ref_intensities) == [103.0] * len(LED_IDS)
        assert [str(warning.message) for warning in recwarn] == []

    def test_unreadable_first_image_raises_instead_of_leaving_the_reference_empty(self):
        # The reported case: image 1 is the only reference image and could not be read, which used to
        # leave the reference intensities empty without a word
        solver = make_solver([2, 3, 4], num_ref_imgs=1)

        with pytest.raises(ValueError, match=r'\[1\]'):
            solver.calc_and_set_ref_intensities()

    def test_unreadable_first_image_of_a_larger_window_warns(self):
        solver = make_solver([2, 3], num_ref_imgs=3)

        with pytest.warns(UserWarning, match=r'\[1\]'):
            solver.calc_and_set_ref_intensities()

        # The remaining images of the window still give reference intensities
        assert list(solver.ref_intensities) == [100.0] * len(LED_IDS)

    def test_window_reaching_past_the_analysed_images_warns(self):
        solver = make_solver([1, 2], num_ref_imgs=4)

        with pytest.warns(UserWarning, match=r'\[3, 4\]'):
            solver.calc_and_set_ref_intensities()

        assert list(solver.ref_intensities) == [100.0] * len(LED_IDS)

    def test_raised_message_names_the_settings_to_change(self):
        solver = make_solver([2, 3], num_ref_imgs=1)

        with pytest.raises(ValueError, match='num_ref_images or ref_img_indices'):
            solver.calc_and_set_ref_intensities()
