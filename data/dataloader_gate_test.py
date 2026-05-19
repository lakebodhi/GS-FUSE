"""
Dataloader for gate openness tests only.
Implements an event set with 60/20/20 split by DATE (sort events by time first),
so the test set is the middle 20% by time and covers all event types.
Does not modify data.dataloader; this is a separate implementation.
"""

from collections import defaultdict
import glob
import os
import random
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch import from_numpy
from torch.utils.data import DataLoader

from data.dataloader import data_set_new


class event_set_date_split(object):
    """
    Same as event_set in data.dataloader, but event_files are SORTED BY DATE
    before building self.data, so train/vali/test are 60/20/20 by time
    (test = middle 20% chronologically, all event types present).
    """

    def __init__(
        self,
        seq_len,
        pred_len,
        event_id=0,
        series_id="SP500",
        shuffle=False,
        batch_size=32,
        event_dir="data/event",
        series_dir="data/series",
        scale=True,
    ):
        self.seq_len = seq_len
        self.pred_len = pred_len

        event_dirs = []
        for main_folder in os.listdir(event_dir):
            main_folder_path = os.path.join(event_dir, main_folder)
            if os.path.isdir(main_folder_path):
                for sub_folder in os.listdir(main_folder_path):
                    sub_folder_path = os.path.join(main_folder_path, sub_folder)
                    if os.path.isdir(sub_folder_path):
                        event_dirs.append(sub_folder_path)

        series_file = None
        for file in os.listdir(series_dir):
            if file.split(".")[0] == series_id:
                series_file = os.path.join(series_dir, file)
                break

        if series_file is None:
            print("No time series data found")
            exit(0)
        else:
            raw_df = pd.read_csv(series_file)

        self.dates = raw_df["date_int"].tolist()

        event_files = []
        for event_folder in event_dirs:
            for file in os.listdir(event_folder):
                event_type = os.path.basename(os.path.dirname(event_folder))
                if file.endswith("txt_full_summary.txt"):
                    date = file.split(".")[0]
                    if event_id == 4:
                        date += "1400005"
                    else:
                        date += "083000"

                    if int(date) > self.dates[0] and int(date) < self.dates[-1]:
                        full_summary_path = os.path.join(event_folder, file)
                        sent_reports = []
                        for i in range(11):
                            pattern = os.path.join(event_folder, "????????.txt_report_sent{}.txt".format(i))
                            matches = glob.glob(pattern)
                            if matches:
                                sent_path = max(matches, key=os.path.getmtime)
                                sent_reports.append(sent_path)
                        if len(sent_reports) != 10:
                            sent_reports = sent_reports[:10]
                        nagative_samples_on_type = self._negative_event_based_on_type(
                            date, event_type, event_folder, n=5
                        )
                        event_files.append([date, full_summary_path, sent_reports, nagative_samples_on_type, event_type])

        if len(event_files) < 10:
            print("Not enough event data found")
            exit(0)

        # NEW TEST SET: sort by date so 60/20/20 is time-based (test = middle 20%)
        event_files.sort(key=lambda x: int(x[0]))

        self.data = []
        for date, full_summary, sent_reports, nagative_samples_on_type, event_type in event_files:
            id = self._b_search(int(date), self.dates)
            # Same filter as check_test_set_event_types: idx + pred_len <= len(dates)
            if id < seq_len - 1 or id + pred_len > len(self.dates):
                continue
            else:
                seq_data = from_numpy(
                    raw_df.iloc[id - seq_len + 1 : id + 1].drop(columns=["date", "date_int"]).to_numpy()
                )
                pred_data = from_numpy(
                    raw_df.iloc[id + 1 : id + pred_len + 1].drop(columns=["date", "date_int"]).to_numpy()
                )
                self.data.append(
                    [full_summary, sent_reports, nagative_samples_on_type, seq_data, pred_data, event_type]
                )

        if shuffle:
            random.shuffle(self.data)

        self.d = raw_df.shape[1] - 2
        ratio = [0.6, 0.2, 0.2]
        n = len(self.data)
        border1 = [0, int(ratio[0] * n), int((ratio[0] + ratio[1]) * n)]
        border2 = [int(ratio[0] * n), int((ratio[0] + ratio[1]) * n), n]

        self.scaler = StandardScaler()
        if scale:
            train_seq_data = np.concatenate(
                [i[3].numpy() for i in self.data[border1[0] : border2[0]]]
                + [i[4].numpy() for i in self.data[border1[0] : border2[0]]],
                axis=0,
            )
            self.scaler.fit(train_seq_data)
            for data in self.data:
                data.append(from_numpy(self.scaler.transform(data[3])))
                data.append(from_numpy(self.scaler.transform(data[4])))

        self.train_set, self.train_loader = self._get_data(self.data[border1[0] : border2[0]], batch_size)
        self.test_set, self.test_loader = self._get_data(self.data[border1[1] : border2[1]], batch_size)
        self.vali_set, self.vali_loader = self._get_data(self.data[border1[2] : border2[2]], batch_size)
        print(
            "Data Loaded (date-split test set): TRAIN: {}, VALI: {}, TEST: {}".format(
                len(self.train_set), len(self.vali_set), len(self.test_set)
            )
        )

    def _b_search(self, date, dates):
        i, j = 0, len(dates) - 1
        while i <= j:
            mid = (i + j) // 2
            if mid >= len(dates) - 1:
                break
            if date >= dates[mid] and date < dates[mid + 1]:
                return mid
            else:
                if date < dates[mid]:
                    j = mid - 1
                if date >= dates[mid + 1]:
                    i = mid + 1
        return -1

    def _get_data(self, data, batch_size, num_workers=2, prefetch_factor=1):
        # Reduced num_workers from 15 to 2 and prefetch_factor to 1 to minimize memory consumption
        # Each worker process loads data into memory, and prefetch_factor multiplies that
        # With num_workers=2 and prefetch_factor=1, we only have 2 batches prefetched total
        dataset = data_set_new(data)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor if num_workers > 0 else None,
            pin_memory=False  # Disable pin_memory to reduce memory usage
        )
        return dataset, dataloader

    def _negative_event_based_on_type(self, date, current_event_type, event_folder, n=5):
        date = date[:8]
        event_folders = []
        event_list = ["1", "2", "3", "4", "5", "6"]
        base_path = os.path.dirname(os.path.dirname(event_folder))

        for event_type in event_list:
            if event_type != current_event_type:
                folder_path = os.path.join(base_path, str(event_type))
                event_folders.append(folder_path)

        event_files = defaultdict(list)
        target_date = datetime.strptime(date, "%Y%m%d")

        for folder in event_folders:
            event_type = os.path.basename(folder)
            closest_folders = self._find_closest_n_dates(target_date, folder, n)
            for closest_folder in closest_folders:
                if closest_folder:
                    full_path = os.path.join(folder, closest_folder)
                    for root, dirs, files in os.walk(full_path):
                        for file in files:
                            if file.endswith("txt_full_summary.txt"):
                                summary_file = os.path.join(root, file)
                                if os.path.exists(summary_file):
                                    event_files[event_type].append(summary_file)

        summary_files = []
        event_types_with_files = list(event_files.keys())
        while len(summary_files) < n and event_types_with_files:
            random.shuffle(event_types_with_files)
            for event_type in event_types_with_files[:]:
                if event_files[event_type]:
                    summary_files.append(event_files[event_type].pop())
                    if len(summary_files) == n:
                        break
                if not event_files[event_type]:
                    event_types_with_files.remove(event_type)
        return summary_files

    def _find_closest_n_dates(self, target_date, event_folder, n=6):
        subfolders = [f for f in os.listdir(event_folder) if os.path.isdir(os.path.join(event_folder, f))]
        if not subfolders:
            return []
        sorted_dates = sorted(subfolders)
        target_date_str = target_date.strftime("%Y%m%d")
        closest_index = self._b_search_str(target_date_str, sorted_dates)
        left = closest_index - 1
        right = closest_index + 1
        left_dates = []
        right_dates = []
        while len(left_dates) < n // 2 and left >= 0:
            left_dates.append(sorted_dates[left])
            left -= 1
        while len(right_dates) < n // 2 and right < len(sorted_dates):
            right_dates.append(sorted_dates[right])
            right += 1
        while len(left_dates) + len(right_dates) < n and left >= 0:
            left_dates.append(sorted_dates[left])
            left -= 1
        while len(left_dates) + len(right_dates) < n and right < len(sorted_dates):
            right_dates.append(sorted_dates[right])
            right += 1
        return sorted(left_dates + right_dates)

    def _b_search_str(self, date_str, sorted_dates):
        """Binary search for date string in list of folder names (e.g. 20021017.txt)."""
        for i, d in enumerate(sorted_dates):
            if date_str <= d.replace(".txt", ""):
                return min(i, len(sorted_dates) - 1)
        return len(sorted_dates) - 1


def event_set_gate_test(
    seq_len,
    pred_len,
    event_id=0,
    series_id="SP500",
    shuffle=False,
    batch_size=32,
    event_dir="data/event",
    series_dir="data/series",
    scale=True,
):
    """Event set with test split by date (middle 20% by time). Use for gate openness scripts only."""
    return event_set_date_split(
        seq_len=seq_len,
        pred_len=pred_len,
        event_id=event_id,
        series_id=series_id,
        shuffle=shuffle,
        batch_size=batch_size,
        event_dir=event_dir,
        series_dir=series_dir,
        scale=scale,
    )
