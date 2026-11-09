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



class BinaryDataFiles:
    """Fixtures and helpers shared by the tests that build the binary data of a channel."""

    # led_id, led_array_id, sum_col_val, mean_col_val, max_col_val, bgsub_sum_col_val
    NUM_COLUMNS = 6
    # the same file written by v0.11.0, which adds sidebg_sum_col_val, sidebg_pedestal and bgsub2_sum_col_val
    NUM_COLUMNS_V011 = 9
    NUM_LEDS = 2

    @pytest.fixture(autouse=True)
    def clean_coordinate_cache(self):
        # _append_coordinates caches the coordinate file in a function attribute for the whole process
        from ledsa.core.file_handling import _append_coordinates
        _append_coordinates.__dict__.pop('coord', None)
        yield
        _append_coordinates.__dict__.pop('coord', None)

    @pytest.fixture
    def analysis_dir(self, tmp_path, monkeypatch):
        # create_binary_data only reads the image range out of the config, so a mapping is enough and no
        # config.ini has to be written
        from ledsa.core import file_handling
        monkeypatch.setattr(file_handling, 'ConfigData',
                            lambda: {'DEFAULT': {'num_img_overflow': 'None'},
                                     'analyse_photo': {'first_img_analysis_id': '1',
                                                       'last_img_analysis_id': '2',
                                                       'num_skip_imgs': '0'}})
        channel_dir = tmp_path / 'analysis' / 'channel0'
        channel_dir.mkdir(parents=True)
        coordinates = '\n'.join(f'{float(led_id)},{float(led_id)},{float(led_id) + 0.5}'
                                 for led_id in range(self.NUM_LEDS))
        coordinates_path = tmp_path / 'analysis' / 'led_search_areas_with_coordinates.csv'
        coordinates_path.write_text(f'{coordinates}\n', encoding='utf-8')
        monkeypatch.chdir(tmp_path)
        return tmp_path

    def _write_result_file(self, tmp_path, img_id, num_columns, offset=0):
        rows = '\n'.join(','.join(str(float(led_id * 10 + column + offset)) for column in range(num_columns))
                          for led_id in range(self.NUM_LEDS))
        path = tmp_path / 'analysis' / 'channel0' / f'{img_id}_led_positions.csv'
        path.write_text(f'# header\n{rows}\n', encoding='utf-8')
        return path

    def _load(self, tmp_path):
        import pandas as pd
        return pd.read_hdf(tmp_path / 'analysis' / 'channel0' / 'all_parameters.h5', key='channel_values')

    def _binary_path(self, tmp_path):
        return tmp_path / 'analysis' / 'channel0' / 'all_parameters.h5'

    def _set_age_against_the_binary(self, tmp_path, path, seconds):
        """Give a source file a modification time relative to the one of the binary data.

        The modification times of two files written one after the other can land in the same clock tick,
        which would make the age of the source depend on the file system. They are set explicitly instead.
        """
        binary_mtime = os.path.getmtime(self._binary_path(tmp_path))
        os.utime(path, (binary_mtime + seconds, binary_mtime + seconds))

    def _make_all_sources_older(self, tmp_path):
        for path in [tmp_path / 'analysis' / 'led_search_areas_with_coordinates.csv',
                     *(tmp_path / 'analysis' / 'channel0').glob('*_led_positions.csv')]:
            self._set_age_against_the_binary(tmp_path, path, -10)


class TestCreateBinaryData(BinaryDataFiles):

    def test_every_led_of_every_image_is_loaded(self, analysis_dir):
        from ledsa.core.file_handling import create_binary_data
        for img_id in (1, 2):
            self._write_result_file(analysis_dir, img_id, self.NUM_COLUMNS)
        create_binary_data(0)
        table = self._load(analysis_dir)
        assert len(table) == 2 * self.NUM_LEDS
        assert sorted(table['img_id'].unique()) == [1, 2]

    def test_a_binary_file_of_an_earlier_run_is_replaced(self, analysis_dir, capsys):
        # appending kept every LED of the earlier run next to the new one, so the rows piled up with
        # every run and the analysis saw each image twice
        from ledsa.core.file_handling import create_binary_data
        for img_id in (1, 2):
            self._write_result_file(analysis_dir, img_id, self.NUM_COLUMNS)
        create_binary_data(0)
        rows_of_the_first_run = len(self._load(analysis_dir))
        create_binary_data(0)
        assert len(self._load(analysis_dir)) == rows_of_the_first_run
        assert 'is replaced' in capsys.readouterr().out

    def test_a_replaced_binary_file_holds_the_new_values(self, analysis_dir):
        from ledsa.core.file_handling import create_binary_data
        for img_id in (1, 2):
            self._write_result_file(analysis_dir, img_id, self.NUM_COLUMNS)
        create_binary_data(0)
        for img_id in (1, 2):
            self._write_result_file(analysis_dir, img_id, self.NUM_COLUMNS)
        rows = self._load(analysis_dir)
        create_binary_data(0)
        assert self._load(analysis_dir)['sum_col_val'].tolist() == rows['sum_col_val'].tolist()

    def test_differing_column_counts_name_the_re_run(self, analysis_dir):
        # step 3 was run again for part of the images with a version that writes further columns
        from ledsa.core.file_handling import create_binary_data
        self._write_result_file(analysis_dir, 1, self.NUM_COLUMNS)
        self._write_result_file(analysis_dir, 2, self.NUM_COLUMNS_V011)
        with pytest.raises(ValueError, match=r'2_led_positions\.csv has 9 columns, while .*'
                                             r'1_led_positions\.csv has 6 columns.*'
                                             r'Run the data extraction'):
            create_binary_data(0)

    def test_differing_column_counts_are_found_against_the_wider_first_file(self, analysis_dir):
        # the names come from the lowest image ID, so the wider file may be the one that is read first
        from ledsa.core.file_handling import create_binary_data
        self._write_result_file(analysis_dir, 1, self.NUM_COLUMNS_V011)
        self._write_result_file(analysis_dir, 2, self.NUM_COLUMNS)
        with pytest.raises(ValueError, match=r'2_led_positions\.csv has 6 columns, while .*'
                                             r'1_led_positions\.csv has 9 columns'):
            create_binary_data(0)

    def _patch_config(self, monkeypatch, first_img_id, last_img_id, num_skip_imgs):
        from ledsa.core import file_handling
        monkeypatch.setattr(file_handling, 'ConfigData',
                            lambda: {'DEFAULT': {'num_img_overflow': 'None'},
                                     'analyse_photo': {'first_img_analysis_id': str(first_img_id),
                                                       'last_img_analysis_id': str(last_img_id),
                                                       'num_skip_imgs': str(num_skip_imgs)}})

    def test_the_last_image_of_a_run_with_skipped_images_is_loaded(self, analysis_dir, monkeypatch):
        # the IDs 1 to 11 with every second image skipped are the six images 1, 3, 5, 7, 9 and 11,
        # whose results step 3 numbers 1 to 6. Rounding the count down left out the last of them,
        # without listing it among the images that hold no results.
        from ledsa.core.file_handling import create_binary_data
        self._patch_config(monkeypatch, first_img_id=1, last_img_id=11, num_skip_imgs=1)
        for img_id in range(1, 7):
            self._write_result_file(analysis_dir, img_id, self.NUM_COLUMNS)
        create_binary_data(0)
        assert sorted(self._load(analysis_dir)['img_id'].unique()) == [1, 2, 3, 4, 5, 6]

    def test_no_image_beyond_the_last_analysed_one_is_looked_for(self, analysis_dir, monkeypatch,
                                                                 capsys):
        # the IDs 1 to 10 with every second image skipped are the five images 1, 3, 5, 7 and 9
        from ledsa.core.file_handling import create_binary_data
        self._patch_config(monkeypatch, first_img_id=1, last_img_id=10, num_skip_imgs=1)
        for img_id in range(1, 6):
            self._write_result_file(analysis_dir, img_id, self.NUM_COLUMNS)
        create_binary_data(0)
        assert sorted(self._load(analysis_dir)['img_id'].unique()) == [1, 2, 3, 4, 5]
        assert 'No analysis results' not in capsys.readouterr().out

    def test_an_image_without_results_is_still_skipped(self, analysis_dir):
        # image 1 could not be read, so it has no result file and the names come from image 2
        from ledsa.core.file_handling import create_binary_data
        self._write_result_file(analysis_dir, 2, self.NUM_COLUMNS)
        create_binary_data(0)
        assert sorted(self._load(analysis_dir)['img_id'].unique()) == [2]


class TestOutdatedBinaryData(BinaryDataFiles):
    """The binary data are a cache of the step 3 result files, so they must not outlive them.

    read_hdf used to answer with whatever all_parameters.h5 held and only built it when it was missing. An
    analysis run after the data extraction had run again therefore reported the extinction coefficients of
    the values of the earlier run, without a word.
    """

    def _write_both_result_files(self, tmp_path, offset=0):
        for img_id in (1, 2):
            self._write_result_file(tmp_path, img_id, self.NUM_COLUMNS, offset=offset)

    def test_a_rerun_of_step_3_is_picked_up(self, analysis_dir):
        from ledsa.core.file_handling import read_hdf
        self._write_both_result_files(analysis_dir)
        values_of_the_first_run = read_hdf(0)['sum_col_val'].tolist()
        self._write_both_result_files(analysis_dir, offset=100)
        for img_id in (1, 2):
            self._set_age_against_the_binary(
                analysis_dir, analysis_dir / 'analysis' / 'channel0' / f'{img_id}_led_positions.csv', 10)
        assert read_hdf(0)['sum_col_val'].tolist() == [value + 100 for value in values_of_the_first_run]

    def test_the_rebuild_is_reported(self, analysis_dir, capsys):
        # the loading time of the result files is paid again, so the run says why
        from ledsa.core.file_handling import read_hdf
        self._write_both_result_files(analysis_dir)
        read_hdf(0)
        self._write_both_result_files(analysis_dir, offset=100)
        for img_id in (1, 2):
            self._set_age_against_the_binary(
                analysis_dir, analysis_dir / 'analysis' / 'channel0' / f'{img_id}_led_positions.csv', 10)
        capsys.readouterr()
        read_hdf(0)
        assert 'is rebuilt' in capsys.readouterr().out

    def test_an_untouched_binary_file_is_not_rebuilt(self, analysis_dir, capsys):
        # the result files of a real experiment take minutes to load, so an unchanged channel is read as it is
        from ledsa.core.file_handling import read_hdf
        self._write_both_result_files(analysis_dir)
        read_hdf(0)
        self._make_all_sources_older(analysis_dir)
        capsys.readouterr()
        read_hdf(0)
        assert 'is rebuilt' not in capsys.readouterr().out

    def test_new_led_coordinates_rebuild_the_binary_data(self, analysis_dir, capsys):
        # the coordinates are appended to every row of the binary data, so step 1 run again outdates them
        from ledsa.core.file_handling import read_hdf
        self._write_both_result_files(analysis_dir)
        read_hdf(0)
        self._make_all_sources_older(analysis_dir)
        coordinates_path = analysis_dir / 'analysis' / 'led_search_areas_with_coordinates.csv'
        self._set_age_against_the_binary(analysis_dir, coordinates_path, 10)
        capsys.readouterr()
        read_hdf(0)
        assert 'is rebuilt' in capsys.readouterr().out

    def test_a_changed_config_rebuilds_the_binary_data(self, analysis_dir, capsys):
        # the image range of config.ini decides which result files end up in the binary data, so a
        # narrowed range outdates them even though no result file was touched
        from ledsa.core.file_handling import read_hdf
        self._write_both_result_files(analysis_dir)
        read_hdf(0)
        self._make_all_sources_older(analysis_dir)
        config_path = analysis_dir / 'config.ini'
        config_path.write_text('[DEFAULT]\n', encoding='utf-8')
        self._set_age_against_the_binary(analysis_dir, config_path, 10)
        capsys.readouterr()
        read_hdf(0)
        assert 'is rebuilt' in capsys.readouterr().out

    def test_the_color_corrected_columns_do_not_rebuild_the_binary_data(self, analysis_dir, capsys):
        # the analysis writes sum_col_val_cc into the binary data itself, which must not look like a
        # change of a source and drop the column again on the next read
        from ledsa.core.file_handling import extend_hdf, read_hdf
        self._write_both_result_files(analysis_dir)
        num_rows = len(read_hdf(0))
        extend_hdf(0, 'sum_col_val_cc', list(range(num_rows)))
        self._make_all_sources_older(analysis_dir)
        capsys.readouterr()
        fit_parameters = read_hdf(0)
        assert 'is rebuilt' not in capsys.readouterr().out
        assert 'sum_col_val_cc' in fit_parameters.columns

    def test_a_missing_binary_file_is_still_built(self, analysis_dir):
        # the comparison must not stand in the way of the first run of a channel
        from ledsa.core.file_handling import read_hdf
        self._write_both_result_files(analysis_dir)
        assert not self._binary_path(analysis_dir).exists()
        assert len(read_hdf(0)) == 2 * self.NUM_LEDS
