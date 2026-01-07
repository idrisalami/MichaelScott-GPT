from typing import List, Tuple
import torch
from torch.utils.data import Dataset

class OfficeSeq2Seq(Dataset):
    """Seq2Seq dataset that pads inside __getitem__ and does teacher forcing shift."""
    def __init__(self,
                 enc_src: List[List[int]],
                 enc_tgt: List[List[int]],
                 max_src: int,
                 max_tgt: int,
                 pad_id: int) -> None:
        super().__init__()
        assert len(enc_src) == len(enc_tgt), "src/tgt length mismatch"
        self.src = enc_src
        self.tgt = enc_tgt
        self.max_src = max_src
        self.max_tgt = max_tgt
        self.pad_id = pad_id

    @staticmethod
    def _pad_to(ids: List[int], L: int, pad_id: int) -> List[int]:
        if len(ids) >= L:
            return ids[:L]
        return ids + [pad_id] * (L - len(ids))

    def __len__(self) -> int:
        return len(self.src)

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # pad source/target
        s = self._pad_to(self.src[i], self.max_src, self.pad_id)        # (S,)
        t = self._pad_to(self.tgt[i], self.max_tgt, self.pad_id)        # (T,)

        # teacher forcing shift
        dec_in = t[:-1]     # (T-1,)
        labels = t[1:]      # (T-1,)

        # convert to long & contiguous tensors
        s = torch.tensor(s, dtype=torch.long).contiguous()
        dec_in = torch.tensor(dec_in, dtype=torch.long).contiguous()
        labels = torch.tensor(labels, dtype=torch.long).contiguous()
        return s, dec_in, labels

def get_local_data(src_path, tgt_path):
    """Read local data files to upload"""
    def read_lines(p):
        return [l.strip() for l in open(p, encoding="utf-8").read().splitlines() if l.strip()]
    
    src_lines = read_lines(src_path)
    tgt_lines = read_lines(tgt_path)
    return src_lines, tgt_lines