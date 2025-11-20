import torch
import torch.nn as nn
import torch.nn.functional as F

from src.core.wnnLutLayer import WNNLUTLayer

class MultiLayerWNN(nn.Module):
    def __init__(
        self,
        in_bits,
        num_classes,
        lut_input_size=6,
        hidden_luts=(2000, 1000),
        tau=1.0,
    ):
        super().__init__()
        self.tau = tau

        layers = []
        prev_bits = in_bits
        for n_lut in hidden_luts:
            layers.append(
                WNNLUTLayer(
                    in_bits=prev_bits,
                    num_luts=n_lut,
                    lut_input_size=lut_input_size,
                )
            )
            prev_bits = n_lut  # dimension of each layer's output = num_luts

        self.layers = nn.ModuleList(layers)
        self.classifier = nn.Linear(prev_bits, num_classes, bias=False)

        # for pruning variable keep_idx 
        self.register_buffer("keep_idx", None)

    def forward(self, x_bits, return_hidden: bool = False):
        """
        x_bits: [B, in_bits]
        return_hidden: True return (logits, h_last)
        """
        h = x_bits
        for layer in self.layers:
            h = layer(h)  # [B, num_luts]

        # if keep_idx, do hidden dimemsion pruning
        if self.keep_idx is not None:
            h_used = h[:, self.keep_idx]
        else:
            h_used = h

        logits = self.classifier(h_used) / self.tau

        if return_hidden:
            return logits, h
        else:
            return logits