from typing import Tuple, Optional, Dict
import torch

class Masks:
    """
    Utilities to build boolean attention masks for Transformer models.
    Returns masks where True = keep/attend, False = masked-out.

    Usage:
        m = Masks(pad_id=0)
        src_mask = m.encoder(src)                     # (B,1,1,S)
        tgt_pad, tgt_caus, tgt_mask = m.decoder(tgt)  # (B,1,1,T), (1,1,T,T), (B,1,T,T)
    """

    def __init__(self, pad_id: int) -> None:
        self.pad_id = pad_id
        # cache for causal masks keyed by (T, device)
        self._causal_cache: Dict[tuple[int, torch.device], torch.Tensor] = {}

    # ---------- encoder ----------
    def encoder(self, src: torch.Tensor) -> torch.Tensor:
        """
        src: LongTensor (B, S)
        returns: BoolTensor (B, 1, 1, S)
        """
        if src.dim() != 2:
            raise ValueError(f"encoder mask expects src of shape (B,S), got {tuple(src.shape)}")
        return (src != self.pad_id).unsqueeze(1).unsqueeze(2)

    # ---------- decoder ----------
    def decoder(self, tgt_in: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        tgt_in: LongTensor (B, T)
        returns:
            tgt_pad   (B, 1, 1, T)  True at non-PAD
            tgt_caus  (1, 1, T, T)  True where i >= j
            tgt_mask  (B, 1, T, T)  tgt_pad AND tgt_caus
        """
        if tgt_in.dim() != 2:
            raise ValueError(f"decoder mask expects tgt_in of shape (B,T), got {tuple(tgt_in.shape)}")
        B, T = tgt_in.shape
        device = tgt_in.device

        tgt_pad = (tgt_in != self.pad_id).unsqueeze(1).unsqueeze(2)   # (B,1,1,T)
        tgt_caus = self._causal(T, device)                            # (1,1,T,T)
        tgt_mask = tgt_pad & tgt_caus                                 # (B,1,T,T)
        return tgt_pad, tgt_caus, tgt_mask

    # ---------- internals ----------
    def _causal(self, T: int, device: Optional[torch.device] = None) -> torch.Tensor:
        """
        Build or fetch a cached lower-triangular BoolTensor (1,1,T,T).
        """
        if T < 1:
            raise ValueError("T must be >= 1")
        key = (T, device or torch.device("cpu"))
        m = self._causal_cache.get(key)
        if m is None:
            m = torch.tril(torch.ones(T, T, dtype=torch.bool, device=device)).unsqueeze(0).unsqueeze(1)
            self._causal_cache[key] = m
        return m