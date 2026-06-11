import torch

from src.data.dataset import create_src_mask
from src.inference.beam_search import BeamSearch


class _DummyEncoder(torch.nn.Module):
    def forward(self, src_tokens, src_mask=None):
        return torch.zeros((src_tokens.size(0), 1, 4), device=src_tokens.device)


class _DummyDecoder(torch.nn.Module):
    def forward(self, tgt, enc_output, src_mask=None, tgt_mask=None):
        batch_size, tgt_len = tgt.shape
        hidden = torch.zeros((batch_size, tgt_len, 4), device=tgt.device)
        hidden[:, -1, 0] = (tgt[:, -1].eq(2)).float()
        hidden[:, -1, 1] = (tgt[:, -1].eq(4)).float()
        return hidden


class _DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = _DummyEncoder()
        self.decoder = _DummyDecoder()
        self.generator = torch.nn.Linear(4, 6, bias=False)
        with torch.no_grad():
            self.generator.weight.zero_()
            self.generator.weight[4, 0] = 4.0
            self.generator.weight[3, 1] = 5.0


def test_beam_search_stops_on_eos():
    model = _DummyModel()
    search = BeamSearch(model=model, beam_size=2, alpha=0.6, max_len=6, device="cpu")
    src = torch.tensor([[2, 5, 0]], dtype=torch.long)
    src_mask = create_src_mask(src)

    result = search.search(src, src_mask=src_mask)

    assert result == [4]
