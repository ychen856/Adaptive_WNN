import torch
import torch.nn as nn
import torch.nn.functional as F

class WNNLUTLayer(nn.Module):
    def __init__(self, in_bits, num_luts, lut_input_size=6, conn_idx=None, init_std=0.01):
        super().__init__()   

        self.in_bits = in_bits
        self.num_luts = num_luts
        self.lut_input_size = lut_input_size

        # randomly or externally given conn_idx
        if conn_idx is None:
            conn = torch.randint(
                low=0,
                high=in_bits,
                size=(num_luts, lut_input_size),
                dtype=torch.long,
            )
        else:
            conn = conn_idx.clone().long()

        # only here can register buffer
        self.register_buffer("conn_idx", conn)

        # initialize LUT table
        table = torch.zeros(num_luts, 2 ** lut_input_size)
        table = table.normal_(mean=0.0, std=init_std)
        self.table = nn.Parameter(table)

        # powers for bit → index
        powers = (2 ** torch.arange(lut_input_size)).float()
        self.register_buffer("powers", powers.view(1, 1, -1))

    def forward(self, x_bits):
        """
        x_bits: [B, in_bits], expect to be 0/1
        return: [B, num_luts]
        """
        B = x_bits.size(0)
        device = x_bits.device

        # force to be 0/1
        x_bits = (x_bits > 0.5).float()

        # extract each LUT's corresponding k bits
        # conn_idx: [num_luts, k]
        # -> [B, num_luts, k]
        bits = x_bits[:, self.conn_idx.view(-1)].view(
            B, self.num_luts, self.lut_input_size
        )

        # idx = (((b0)*2 + b1)*2 + b2)*2 + ...
        idx = torch.zeros(B, self.num_luts, dtype=torch.long, device=device)
        for j in range(self.lut_input_size):
            idx = idx * 2 + bits[:, :, j].long()

        # LUT：table: [num_luts, 2^k]
        table_expanded = self.table.unsqueeze(0).expand(B, -1, -1)  # [B, num_luts, 2^k]
        out = torch.gather(table_expanded, 2, idx.unsqueeze(-1)).squeeze(-1)

        # sigmoid
        out = torch.sigmoid(out)
        return out