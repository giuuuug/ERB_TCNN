import sys
import torch
from erb_tcnn_stream import ERBTCNN_Stream

def test():
    model = ERBTCNN_Stream(
        in_channels=257,
        n_blocks=2,
        num_layers=5,
        kernel_size=3,
        init_dilation=2,
        erb_subband_1=65,
        erb_subband_2=64,
        mask_activation="sigmoid"
    )
    
    model.eval()
    
    batch_size = 1
    in_channels = 257
    seq_len = 1
    
    states = model.get_initial_states(batch_size=batch_size, device='cpu')
    
    # NOW TEST WITH T>1 (FAST PATH)
    print("Testing FAST PATH (T=30)...")
    states = model.get_initial_states(batch_size=batch_size, device='cpu')
    x = torch.randn(batch_size, in_channels, 30)
    out, states = model(x, states)
    assert out.shape == (batch_size, in_channels, 30), f"Shape mismatch: {out.shape}"
    print(f"Fast Path: Output shape {out.shape} -> OK")

    print("Checking equivalence between sequential vs block processing (T=10)...")
    torch.manual_seed(42)
    x_full = torch.randn(batch_size, in_channels, 10)
    
    # Block path
    states_block = model.get_initial_states(batch_size=batch_size, device='cpu')
    out_block, _ = model(x_full, states_block)
    
    # Seq path
    states_seq = model.get_initial_states(batch_size=batch_size, device='cpu')
    out_seq_list = []
    for t in range(x_full.shape[2]):
        out_seq, states_seq = model(x_full[:, :, t:t+1], states_seq)
        out_seq_list.append(out_seq)
    out_seq_cat = torch.cat(out_seq_list, dim=2)
    
    diff = (out_block - out_seq_cat).abs().max()
    print(f"Max diff between sequential and block processing: {diff}")
    assert diff < 1e-4, f"Difference is too large: {diff}"
    print("Equivalence confirmed.")

if __name__ == "__main__":
    test()
