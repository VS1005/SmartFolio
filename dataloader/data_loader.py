import os
import sys
import pickle
from torch.utils import data
import torch

class AllGraphDataSampler(data.Dataset):
    def __init__(self, base_dir, gname_list=None,
                 data_start=None, data_middle=None, data_end=None,
                 train_start_date=None, train_end_date=None,
                 val_start_date=None, val_end_date=None,
                 test_start_date=None, test_end_date=None,
                 idx=False, date=True,
                 mode="train"):
        self.data_dir = os.path.join(base_dir)
        self.mode = mode
        self.data_start = data_start
        self.data_middle = data_middle
        self.data_end = data_end
        if gname_list is None:
            self.gnames_all = os.listdir(self.data_dir)
            self.gnames_all.sort()
        if idx:
            if mode == "train":
                self.gnames_all = self.gnames_all[self.data_start:self.data_middle]
            elif mode == "val":
                self.gnames_all = self.gnames_all[self.data_middle:self.data_end]
            elif mode == "test":
                self.gnames_all = self.gnames_all[self.data_end:]
        if date:
            def _safe_slice(start_date, end_date):
                si = self.date_to_idx(start_date)
                ei = self.date_to_idx(end_date)
                if si is None:
                    si = 0
                if ei is None:
                    ei = len(self.gnames_all) - 1
                if ei < si:
                    ei = si
                return self.gnames_all[si:ei + 1]
            if mode == "train":
                self.gnames_all = _safe_slice(train_start_date, train_end_date)
            elif mode == "val":
                self.gnames_all = _safe_slice(val_start_date, val_end_date)
            elif mode == "test":
                self.gnames_all = _safe_slice(test_start_date, test_end_date)
        self.data_all = self.load_state()

    def __len__(self):
        return len(self.data_all)

    def load_state(self):
        data_all = []
        length = len(self.gnames_all)
        for i in range(length):
            sys.stdout.flush()
            sys.stdout.write('{} data loading: {:.2f}%{}'.format(self.mode, i*100/length, '\r'))
            try:
                item = pickle.load(open(os.path.join(self.data_dir, self.gnames_all[i]), "rb"))
            except Exception as e:
                print(f"\nWarning: failed to load {self.gnames_all[i]}: {e}. Skipping.")
                continue

            # Filter out empty or inconsistent items (e.g., 0 stocks)
            try:
                feats = item.get('features', None)
                labels = item.get('labels', None)
                ts_feats = item.get('ts_features', None)
                valid = True
                if isinstance(feats, torch.Tensor):
                    n = feats.shape[0]
                else:
                    n = feats.shape[0] if feats is not None else 0
                if n is None or n == 0:
                    valid = False
                if isinstance(labels, torch.Tensor):
                    if labels.shape[0] != n:
                        valid = False
                elif labels is None:
                    valid = False
                if isinstance(ts_feats, torch.Tensor):
                    if ts_feats.shape[0] != n:
                        valid = False
                elif ts_feats is None:
                    valid = False
                if not valid:
                    print(f"\nSkipping {self.gnames_all[i]} due to empty or inconsistent shapes (n={n}).")
                    continue
            except Exception as e:
                print(f"\nWarning: sanity check failed for {self.gnames_all[i]}: {e}. Skipping.")
                continue

            data_all.append(item)
        print('{} data loaded!'.format(self.mode))
        return data_all

    def __getitem__(self, idx):
        return self.data_all[idx]

    def date_to_idx(self, date):
        result = None
        for i in range(len(self.gnames_all)):
            if date == self.gnames_all[i][:10]:
                result = i
        return result
