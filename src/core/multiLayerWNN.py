import torch
import torch.nn as nn
import torch.nn.functional as F

from src.core.wnnLutLayer import WNNLUTLayer

class MultiLayerWNN(nn.Module):
    def __init__(
        self,
        in_bits: int,
        num_classes: int,
        lut_input_size: int = 6,
        hidden_luts=(2000, 1000),
        tau: float = 1.0,
    ):
        super().__init__()
        self.tau = tau

        layers = []
        prev_bits = in_bits

        self.layer_in_bits = []   # input bits per layer
        self.layer_out_luts = []  # number of LUTs per layer

        for n_lut in hidden_luts:
            layers.append(
                WNNLUTLayer(
                    in_bits=prev_bits,
                    num_luts=n_lut,
                    lut_input_size=lut_input_size,
                )
            )
            self.layer_in_bits.append(prev_bits)
            self.layer_out_luts.append(n_lut)
            prev_bits = n_lut

        self.layers = nn.ModuleList(layers)
        self.classifier = nn.Linear(prev_bits, num_classes, bias=False)

        # for hidden pruning
        self.register_buffer("keep_idx", None)

    def forward(self, x_bits: torch.Tensor) -> torch.Tensor:
        """
        x_bits: [B, in_bits]
        """
        h = x_bits
        for layer in self.layers:
            h = layer(h)  # [B, num_luts_l]

        if self.keep_idx is not None:
            h_used = h[:, self.keep_idx]
        else:
            h_used = h

        logits = self.classifier(h_used) / self.tau
        return logits

    def forward_with_all_hidden(self, x_bits: torch.Tensor):
        """
        Return:
          logits: [B, C]
          h_list: list of length L, where the l-th element is [B, num_luts_l]
        """
        h_list = []
        h = x_bits
        for layer in self.layers:
            h = layer(h)
            h_list.append(h)
        if self.keep_idx is not None:
            h_used = h[:, self.keep_idx]
        else:
            h_used = h
        logits = self.classifier(h_used) / self.tau
        return logits, h_list