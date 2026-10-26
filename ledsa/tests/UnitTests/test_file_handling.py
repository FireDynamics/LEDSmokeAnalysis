import os

import pytest

from ledsa.core.file_handling import _find_first_result_file, _get_column_names


class TestFindFirstResultFile:

    def _write_result_files(self, tmp_path, img_ids):
        channel_dir = tmp_path / 'analysis' / 'channel0'
        channel_dir.mkdir(parents=True)
        for img_id in img_ids:
            (channel_dir / f'{img_id}_led_positions.csv').write_text('# header\n', encoding='utf-8')

    def test_lowest_image_id_wins(self, tmp_path, monkeypatch):
        # the IDs are compared as numbers, so 2 comes before 10
        self._write_result_files(tmp_path, [10, 2, 3])
        monkeypatch.chdir(tmp_path)
        assert _find_first_result_file(0) == os.path.join('analysis', 'channel0', '2_led_positions.csv')

    def test_image_without_results_is_skipped(self, tmp_path, monkeypatch):
        # image 1 could not be read, so it has no result file at all
        self._write_result_files(tmp_path, [2, 4])
        monkeypatch.chdir(tmp_path)
        assert _find_first_result_file(0) == os.path.join('analysis', 'channel0', '2_led_positions.csv')

    def test_other_files_are_ignored(self, tmp_path, monkeypatch):
        self._write_result_files(tmp_path, [3])
        (tmp_path / 'analysis' / 'channel0' / 'all_parameters.h5').write_text('', encoding='utf-8')
        monkeypatch.chdir(tmp_path)
        assert _find_first_result_file(0) == os.path.join('analysis', 'channel0', '3_led_positions.csv')

    def test_channel_without_results_raises(self, tmp_path, monkeypatch):
        self._write_result_files(tmp_path, [])
        monkeypatch.chdir(tmp_path)
        with pytest.raises(FileNotFoundError, match='Run the data extraction'):
            _find_first_result_file(0)

    def test_missing_channel_directory_raises(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        with pytest.raises(FileNotFoundError, match='No analysis results'):
            _find_first_result_file(0)


class TestColumnNamesWithoutTheFirstImage:

    def test_names_are_read_from_the_first_existing_file(self, tmp_path, monkeypatch):
        # the reported case: image 1 could not be read, so the analysis has to fall back to image 2
        channel_dir = tmp_path / 'analysis' / 'channel0'
        channel_dir.mkdir(parents=True)
        row = ','.join(str(float(i)) for i in range(6))
        (channel_dir / '2_led_positions.csv').write_text(f'# header\n{row}\n{row}\n', encoding='utf-8')
        monkeypatch.chdir(tmp_path)
        assert _get_column_names(0) == ["img_id", "led_id", "led_array_id", "sum_col_val", "mean_col_val",
                                        "max_col_val", "bgsub_sum_col_val", "width", "height"]
