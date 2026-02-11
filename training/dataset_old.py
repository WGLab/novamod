import torch
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

class LargeIterableDataset(IterableDataset):
    def __init__(self, file_list, max_samples=None, compute_len=False):
        super().__init__()
        self.file_list = list(file_list)
        self.max_samples = max_samples
        self._length = None
        if compute_len:
            self._length = self._count_windows()

    def _count_windows(self):
        count = 0
        for fname in self.file_list:
            data = torch.load(fname, map_location="cpu")
            n = len(data[0]) if hasattr(data[0], "__len__") else data[0].shape[0]
            count += n
            if self.max_samples and count >= self.max_samples:
                return self.max_samples
        return count

    def __iter__(self):
        info = get_worker_info()
        if info is None:
            files = self.file_list
        else:
            # shard files across workers
            files = self.file_list[info.id::info.num_workers]

        emitted = 0
        for fname in files:
            data = torch.load(fname, map_location="cpu")
            x_all, lb_all, pos_all = data[0], data[1], data[2]

            n = len(x_all)
            for i in range(n):
                yield (x_all[i].float(), lb_all[i], pos_all[i])
                emitted += 1
                if self.max_samples and emitted >= self.max_samples:
                    return

    def __len__(self):
        if self._length is None:
            raise TypeError("Length unknown; set compute_len=True to pre-count.")
        return self._length
