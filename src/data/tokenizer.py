from pathlib import Path
from typing import Iterable

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer

PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
BOS_TOKEN = "<bos>"
EOS_TOKEN = "<eos>"

PAD_ID = 0
UNK_ID = 1
BOS_ID = 2
EOS_ID = 3


class SharedBPETokenizer:
    def __init__(self, vocab_size: int = 32_000) -> None:
        self.vocab_size = vocab_size
        self._tokenizer = Tokenizer(BPE(unk_token=UNK_TOKEN))
        self._tokenizer.pre_tokenizer = Whitespace()
        self._trained = False

    @property
    def tokenizer(self) -> Tokenizer:
        return self._tokenizer

    @property
    def pad_id(self) -> int:
        return PAD_ID

    @property
    def unk_id(self) -> int:
        return UNK_ID

    @property
    def bos_id(self) -> int:
        return BOS_ID

    @property
    def eos_id(self) -> int:
        return EOS_ID

    def train(self, files: Iterable[str | Path]) -> None:
        trainer = BpeTrainer(
            vocab_size=self.vocab_size,
            special_tokens=[PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN],
            show_progress=False,
        )
        file_paths = [str(Path(file_path)) for file_path in files]
        self._tokenizer.train(files=file_paths, trainer=trainer)
        self._validate_special_token_ids()
        self._trained = True

    def _validate_special_token_ids(self) -> None:
        expected_ids = {
            PAD_TOKEN: PAD_ID,
            UNK_TOKEN: UNK_ID,
            BOS_TOKEN: BOS_ID,
            EOS_TOKEN: EOS_ID,
        }
        for token, expected_id in expected_ids.items():
            actual_id = self._tokenizer.token_to_id(token)
            if actual_id != expected_id:
                raise ValueError(
                    f"Special token {token!r} has id {actual_id}, expected {expected_id}"
                )

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        if not self._trained:
            raise RuntimeError("Tokenizer must be trained before encoding text")

        encoding = self._tokenizer.encode(text)
        token_ids = encoding.ids
        if add_special_tokens:
            return [self.bos_id, *token_ids, self.eos_id]
        return token_ids

    def decode(self, ids: Iterable[int]) -> str:
        if not self._trained:
            raise RuntimeError("Tokenizer must be trained before decoding ids")

        return self._tokenizer.decode(list(ids), skip_special_tokens=True)

    def save(self, output_dir: str | Path) -> tuple[Path, Path]:
        if not self._trained:
            raise RuntimeError("Tokenizer must be trained before saving")

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        temp_prefix = "shared_bpe"
        model_vocab_path, model_merges_path = self._tokenizer.model.save(
            str(output_path), temp_prefix
        )

        vocab_path = output_path / "bpe_vocab.json"
        merges_path = output_path / "bpe_merges.txt"

        Path(model_vocab_path).replace(vocab_path)
        Path(model_merges_path).replace(merges_path)

        return vocab_path, merges_path

    @classmethod
    def load(
        cls, vocab_path: str | Path, merges_path: str | Path
    ) -> "SharedBPETokenizer":
        tokenizer = Tokenizer(
            BPE.from_file(str(vocab_path), str(merges_path), unk_token=UNK_TOKEN)
        )
        tokenizer.pre_tokenizer = Whitespace()

        instance = cls()
        instance._tokenizer = tokenizer
        instance._trained = True
        instance._validate_special_token_ids()
        return instance
