# some functions for doing calculations with time strings
# -> make a new class or use time class together with dates


def sub_times(time1, time2):
    t1 = time1.split(':')
    t2 = time2.split(':')
    for i in range(3):
        t1[i] = int(t1[i])
        t2[i] = int(t2[i])
    if t1[0] > t2[0] or (t1[0] == t2[0] and t1[1] > t2[1]) or (t1[0] == t2[0] and t1[1] == t2[1] and t1[2] >= t2[2]):
        s = (t1[2] - t2[2]) % 60
        s_r = (t1[2] - t2[2]) // 60
        m = (t1[1] - t2[1] + s_r) % 60
        m_r = (t1[1] - t2[1] + s_r) // 60
        h = t1[0] - t2[0] + m_r
        return '{:02d}:{:02d}:{:02d}'.format(h, m, s)
    else:
        return '-' + sub_times(time2, time1)


def time_diff(time1, time2):
    t1 = time1.split(':')
    t2 = time2.split(':')
    return (int(t1[0]) - int(t2[0])) * 3600 + (int(t1[1]) - int(t2[1])) * 60 + int(t1[2]) - int(t2[2])


def add_time_diff(time, diff):
    t = time.split(':')
    t[2] = int(t[2]) - int(diff)
    t[1] = int(t[1]) + t[2] // 60
    t[0] = int(t[0]) + t[1] // 60
    t[2] = t[2] % 60
    t[1] = t[1] % 60
    # don't accounts for change in date
    return '{:02d}:{:02d}:{:02d}'.format(t[0], t[1], t[2])


def time_to_int(time):
    t = time.split(':')
    return int(t[0]) * 3600 + int(t[1]) * 60 + int(t[2])