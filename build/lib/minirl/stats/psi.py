import numpy as np
import collections
import pickle
from datetime import datetime
from datetime import timezone
from datetime import timedelta
import copy

SHA_TZ = timezone(
    timedelta(hours=8),
    name="Asia/Shanghai",
)


def get_bj_day():
    utc_now = datetime.utcnow().replace(tzinfo=timezone.utc) - timedelta(hours=11)
    beijing_now = utc_now.astimezone(SHA_TZ)
    _bj = beijing_now.strftime("%Y-%m-%d")  # 结果显示：'2017-10-07'

    return _bj


def get_bj_date():
    utc_now = datetime.utcnow().replace(tzinfo=timezone.utc) - timedelta(hours=11)
    beijing_now = utc_now.astimezone(SHA_TZ)
    _bj = beijing_now

    return _bj


def get_week_day():
    utc_now = datetime.utcnow().replace(tzinfo=timezone.utc) - timedelta(hours=11)
    beijing_now = utc_now.astimezone(SHA_TZ)

    return beijing_now.weekday()


def calculate_psi(expected, actual, buckettype="bins", buckets=10, axis=0):
    """Calculate the PSI (population stability index) across all variables

    Args:
       expected: numpy matrix of original values
       actual: numpy matrix of new values, same size as expected
       buckettype: type of strategy for creating buckets, bins splits into even splits, quantiles splits into quantile buckets
       buckets: number of quantiles to use in bucketing variables
       axis: axis by which variables are defined, 0 for vertical, 1 for horizontal

    Returns:
       psi_values: ndarray of psi values for each variable

    Author:
       Matthew Burke
       github.com/mwburke
       worksofchart.com
    """

    def psi(expected_array, actual_array, buckets):
        """Calculate the PSI for a single variable

        Args:
           expected_array: numpy array of original values
           actual_array: numpy array of new values, same size as expected
           buckets: number of percentile ranges to bucket the values into

        Returns:
           psi_value: calculated PSI value
        """

        def scale_range(input, min, max):
            input += -(np.min(input))
            input /= np.max(input) / (max - min)
            input += min
            return input

        breakpoints = np.arange(0, buckets + 1) / (buckets) * 100

        if buckettype == "bins":
            breakpoints = scale_range(
                breakpoints, np.min(expected_array), np.max(expected_array)
            )
        elif buckettype == "quantiles":
            breakpoints = np.stack(
                [np.percentile(expected_array, b) for b in breakpoints]
            )

        expected_percents = np.histogram(expected_array, breakpoints)[0] / len(
            expected_array
        )
        actual_percents = np.histogram(actual_array, breakpoints)[0] / len(actual_array)

        def sub_psi(e_perc, a_perc):
            """Calculate the actual PSI value from comparing the values.
            Update the actual value to a very small number if equal to zero
            """
            if a_perc == 0:
                a_perc = 0.0001
            if e_perc == 0:
                e_perc = 0.0001

            value = (e_perc - a_perc) * np.log(e_perc / a_perc)
            return value

        sum_gen = (
            sub_psi(expected_percents[i], actual_percents[i])
            for i in range(0, len(expected_percents))
        )
        psi_value = np.sum(np.fromiter(sum_gen, dtype=float))

        return psi_value

    if len(expected.shape) == 1:
        psi_values = np.empty(len(expected.shape))
    else:
        psi_values = np.empty(expected.shape[axis])

    for i in range(0, len(psi_values)):
        if len(psi_values) == 1:
            psi_values = psi(expected, actual, buckets)
        elif axis == 0:
            psi_values[i] = psi(expected[:, i], actual[:, i], buckets)
        elif axis == 1:
            psi_values[i] = psi(expected[i, :], actual[i, :], buckets)

    return psi_values


class OnlinePSI:
    def __init__(
        self,
        window_size=30,
        stat_size=7,
        model_db=None,
        his_db=None,
        buckettype="bins",
        buckets=10,
    ) -> None:
        self.window_size = window_size
        self.stat_size = stat_size
        self._model_db = model_db
        self._his_db = his_db
        self.buckettype = buckettype
        self.buckets = buckets

        self._init_model()

    def _init_model(self):
        self.window = collections.deque(maxlen=self.window_size)
        today = get_bj_day()
        self.psi = {"date": today, "psi": -1}

    def get_defu_psi(self):
        today = get_bj_day()
        return {"date": today, "psi": -1}

    def act(self, model_id, feat_list=[], feat_type="spin"):
        new_model_id = f"{model_id}:{feat_type}"
        # model_key = self.get_model_key(new_model_id)
        old_model = self.get_model(new_model_id)
        if len(feat_list) != 7:
            return self.get_model(new_model_id)
        last7_dates = self.get_dates(start=7, step=7)
        # list 和 dates 一一对应
        lst = feat_list
        data_dict = dict(zip(last7_dates, lst))

        last14_dates = self.get_dates(start=14, step=7)
        old_data_key = self.get_data_key(new_model_id)
        old_data = self._his_db.get(old_data_key)
        tmp_data_key = self.get_data_key(f"{new_model_id}:tmp")
        tmp_data = self._his_db.get(tmp_data_key)
        if old_data is None:
            old_expected_data_dict = data_dict
            self._his_db.set(old_data_key, pickle.dumps(old_expected_data_dict))
            self._his_db.set(tmp_data_key, pickle.dumps(old_expected_data_dict))

            return self.get_defu_psi()
        else:
            old_expected_data_dict = pickle.loads(old_data)
            if tmp_data is None:
                tmp_data = {}
            else:
                tmp_data = pickle.loads(tmp_data)

            expected_data_dict = {}
            for d in last14_dates:
                d1 = old_expected_data_dict.get(d)
                d2 = tmp_data.get(d)
                if d1 is not None:
                    expected_data_dict[d] = d1
                else:
                    if d2 is not None:
                        expected_data_dict[d] = d2
                    else:
                        expected_data_dict[d] = 0
            expected_data_list = list(expected_data_dict.values())
            if sum(expected_data_list) == 0:
                expected_data_list[-1] = 0.00001
            expected_data_array = np.array(expected_data_list)
            actual_array = np.array(feat_list)

            new_psi = calculate_psi(
                expected_data_array,
                actual_array,
                buckettype=self.buckettype,
                buckets=self.buckets,
                axis=0,
            )

            self._his_db.set(tmp_data_key, pickle.dumps(data_dict))

            self._his_db.set(old_data_key, pickle.dumps(expected_data_dict))
            model = {"date": get_bj_day(), "psi": new_psi}
            model["old_date"] = old_model.get("date", get_bj_day())
            model["old_psi"] = old_model.get("psi", -1)
            self.save_model(new_model_id, model)

            return new_psi

    def learn(self):
        pass

    def get_data_key(self, model_id):
        return f"{model_id}:his"

    def get_model_key(self, model_id):
        return f"{model_id}:psi"

    def load_model(self, model_id):
        model_key = self.get_model_key(model_id)
        _model = self._model_db.get(model_key)
        if _model is None:
            today = get_bj_day()
            model = {"date": today, "psi": -1}
        else:
            model = pickle.loads(_model)

        return model

    def save_model(self, model_id, model):
        model_key = self.get_model_key(model_id)
        self._model_db.set(model_key, pickle.dumps(model))

    def get_model(self, model_id):
        return self.load_model(model_id)

    def set_model(self, model):
        self.psi = copy.deepcopy(model)

    def get_dates(self, start=14, step=7):
        end_date = get_bj_date()
        start_date = end_date - timedelta(days=start)

        # 生成日期列表
        dates = []
        for i in range(step):
            date = start_date + timedelta(days=i)
            dates.append(date.strftime("%Y-%m-%d"))
        return dates

    def get_tmp_data(self, model_id, feat_type="spin"):
        new_model_id = f"{model_id}:{feat_type}"
        tmp_data_key = self.get_data_key(f"{new_model_id}:tmp")

        tmp_data = self._his_db.get(tmp_data_key)
        if tmp_data is None:
            return {}
        else:
            return pickle.loads(tmp_data)

    def get_base_data(self, model_id, feat_type="spin"):
        new_model_id = f"{model_id}:{feat_type}"
        old_data_key = self.get_data_key(new_model_id)
        data = self._his_db.get(old_data_key)
        if data is None:
            return {}
        else:
            return pickle.loads(data)
