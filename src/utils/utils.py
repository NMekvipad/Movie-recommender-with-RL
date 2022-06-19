import time
import datetime
import pandas as pd


def second_to_hhmmss(seconds):
    return str(datetime.timedelta(seconds=seconds))


class RunTimer:
    def __init__(self):
        self.start_time = time.time()
        self.previous_time = self.start_time
        time_elapse = self.previous_time - self.start_time

        self._time_record = {
            'message_record': ['Start timer'],
            'time_epoch': [self.start_time],
            'time': [second_to_hhmmss(self.start_time)],
            'time_elapse_epoch': [time_elapse],
            'time_elapse': [second_to_hhmmss(time_elapse)],
            'time_elapse_epoch_from_prev': [time_elapse],
            'time_elapse_from_prev': [second_to_hhmmss(time_elapse)],
        }

        self.time_record = pd.DataFrame(self._time_record)

    def get_time_elaspe(self, message='', return_time_elapse=False):
        cur_time = time.time()
        time_from_start = cur_time - self.start_time
        time_from_prev_record = cur_time - self.previous_time

        self.previous_time = cur_time
        self._time_record['message_record'].append(message)
        self._time_record['time_epoch'].append(cur_time)
        self._time_record['time'].append(second_to_hhmmss(cur_time))
        self._time_record['time_elapse_epoch'].append(time_from_start)
        self._time_record['time_elapse'].append(second_to_hhmmss(time_from_start))
        self._time_record['time_elapse_epoch_from_prev'].append(time_from_prev_record)
        self._time_record['time_elapse_from_prev'].append(second_to_hhmmss(time_from_prev_record))
        self.time_record = pd.DataFrame(self._time_record)

        print('{message}: time elapse from start is {time_from_start}'.format(message=message, time_from_start=time_from_start))
        print('{message}: time elapse from previous step is {time_from_prev_record}'.format(message=message, time_from_prev_record=time_from_prev_record))

        if return_time_elapse:
            return {'time_from_start': time_from_start, 'time_from_prev_record': time_from_prev_record}


def join_string(strings):
    strings = [str(s) if s is not None else '' for s in strings]

    return ' '.join(strings)

