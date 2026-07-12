import numpy as np
import pytest

import ledsa.core.image_reading as image_reading
from ledsa.core.image_handling import format_img_name


class TestRawExtensionRouting:
    @pytest.fixture
    def raw_reader_stub(self, monkeypatch):
        calls = []

        def stub(filename, channel):
            calls.append(filename)
            return np.zeros((4, 4))

        monkeypatch.setattr(image_reading, '_read_channel_data_from_raw_file', stub)
        return calls

    @pytest.mark.parametrize('extension', ['.CR2', '.CR3', '.NEF', '.ARW', '.DNG'])
    def test_raw_formats_are_routed_to_raw_reader(self, raw_reader_stub, extension):
        image_reading.read_channel_data_from_img(f'img_0001{extension}', channel=0)
        assert raw_reader_stub == [f'img_0001{extension}']

    def test_unsupported_format_raises(self):
        with pytest.raises(ValueError, match='Unsupported image format'):
            image_reading.read_channel_data_from_img('img_0001.xyz', channel=0)


class TestFormatImgName:
    def test_plain_placeholder_preserves_leading_zeros(self):
        assert format_img_name('DSC_{}.NEF', '0001') == 'DSC_0001.NEF'

    def test_plain_placeholder_with_int_id(self):
        assert format_img_name('test_img_{}.jpg', 7) == 'test_img_7.jpg'

    def test_numeric_format_spec_with_string_id(self):
        assert format_img_name('IMG_{:04d}.CR2', '7') == 'IMG_0007.CR2'

    def test_numeric_format_spec_with_int_id(self):
        assert format_img_name('IMG_{:04d}.CR2', 7) == 'IMG_0007.CR2'
