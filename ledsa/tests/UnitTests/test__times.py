from unittest import TestCase
from ledsa.core import _times


class TestTimeFunctions(TestCase):
    def setUp(self):
        self.time1 = '01:00:00'
        self.time2 = '00:30:00'

    def test_time_to_int(self):
        time = '01:11:22'
        time_int = _times.time_to_int(time)
        self.assertEqual(60*60+11*60+22, time_int)

    def test_sub_times(self):
        diff = _times.sub_times(self.time1, self.time2)
        self.assertEqual('00:30:00', diff)

    def test_sub_times_over_midnight(self):
        diff = _times.sub_times(self.time2, self.time1)
        self.assertEqual('-00:30:00', diff)

    def test_time_diff(self):
        diff = _times.time_diff(self.time1, self.time2)
        self.assertEqual(_times.time_to_int(_times.sub_times(self.time1, self.time2)), diff)

    def test_time_diff_with_neg_time(self):
        diff = _times.time_diff('-' + self.time1, self.time2)
        self.assertEqual(90*60, diff)

    def test_add_time_diff(self):
        new_time = _times.add_time_diff(self.time1, 3666)
        self.assertEqual('02:01:06', new_time)



