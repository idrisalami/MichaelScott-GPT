from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, List, Sequence
import tiktoken

@dataclass(frozen=True)
class VocabInfo:
    pad_id: int = 0
    bos_id: int = 1
    eos_id: int = 2
    shift:  int = 3                      # reserve 0..2 for PAD/BOS/EOS
    @property
    def size(self) -> int:               # no args here; size is “base + shift”
        # For GPT-2 BPE the base is 50257; the tokenizer class will add it.
        # This property represents just the SPECIALS footprint (3).
        return self.shift

class TiktokenTokenizer:
    """
    Minimal tiktoken-based tokenizer (GPT-2 BPE).
    - Encodes strings to BPE IDs, then shifts by +vocab.shift so 0..2 are PAD/BOS/EOS.
    - Decodes by removing specials and unshifting, then using tiktoken's decode.
    """

    def __init__(self, vocab: VocabInfo | None = None, name: str = "gpt2") -> None:
        self.enc = tiktoken.get_encoding(name)          # fixed base vocab (e.g., 50257)
        self.vocab = vocab or VocabInfo()
        # Total vocab available to the model = base + shift
        self.vocab_size = self.enc.n_vocab + self.vocab.shift

    def encode(self, s: str) -> List[int]:
        """Encode a single string to a list of token ids with BOS/EOS and SHIFT applied."""
        base_ids = self.enc.encode(s)                   # in [0, enc.n_vocab-1]
        ids = [i + self.vocab.shift for i in base_ids]  # move up by +3 to free 0..2
        return [self.vocab.bos_id] + ids + [self.vocab.eos_id]

    def decode(self, ids: Sequence[int]) -> str:
        """Decode a sequence of ids back to text (ignores specials)."""
        base = [i - self.vocab.shift for i in ids if i >= self.vocab.shift]
        return self.enc.decode(base)

    # Convenience batch helpers
    def encode_batch(self, texts: Iterable[str]) -> List[List[int]]:
        return [self.encode(t) for t in texts]

    def decode_batch(self, batch_ids: Iterable[Sequence[int]]) -> List[str]:
        return [self.decode(ids) for ids in batch_ids]