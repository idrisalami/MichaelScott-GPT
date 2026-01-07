from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, List, Sequence

@dataclass(frozen=True)
class VocabInfo:
    pad_id: int = 0
    bos_id: int = 1
    eos_id: int = 2
    shift:  int = 3            # reserve 0..2 for PAD/BOS/EOS
    @property
    def size(self) -> int:
        return self.shift

class ByteTokenizer:
    """
    Minimal byte-level tokenizer.
    - Encodes strings to byte IDs with a fixed SHIFT to leave room for specials.
    - Decodes IDs back to UTF-8 strings (invalid sequences are replaced).
    """

    def __init__(self, vocab: VocabInfo | None = None) -> None:
        self.vocab = vocab or VocabInfo()
        # Total vocab available to the model = base + shift
        self.vocab_size = 256 + self.vocab.shift

    @staticmethod
    def _to_bytes(s: str) -> List[int]:
        # Fast UTF-8 encode to raw bytes → list of ints in [0,255]
        return list(s.encode("utf-8"))

    def encode(self, s: str) -> List[int]:
        """Encode a single string to a list of token ids."""
        ids = [b + self.vocab.shift for b in self._to_bytes(s)]
        return [self.vocab.bos_id] + ids + [self.vocab.eos_id]

    def decode(self, ids: Sequence[int]) -> str:
        """Decode a sequence of ids back to text (ignores specials)."""
        bytes_list = [i - self.vocab.shift for i in ids if i >= self.vocab.shift]
        return bytes(bytes_list).decode("utf-8", errors="replace")

    # Convenience batch helpers
    def encode_batch(self, texts: Iterable[str]) -> List[List[int]]:
        return [self.encode(t) for t in texts]

    def decode_batch(self, batch_ids: Iterable[Sequence[int]]) -> List[str]:
        return [self.decode(ids) for ids in batch_ids]