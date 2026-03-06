import torch
import sys
import os

# Adapt path if necessary. Putting script in root of ERB_TCNN
sys.path.append(os.path.join(os.path.dirname(__file__), "speech_enhancement/pt/src/models"))

try:
    from erb_tcnn import ERBTCNN
except ImportError as e:
    print(f"Error importing ERBTCNN: {e}")
    sys.exit(1)

def main():
    print("Instantiating ERBTCNN model...")
    # Use default params (in_channels=257)
    model = ERBTCNN(in_channels=257)
    
    # Put model in eval mode for testing (optional but good practice)
    model.eval()

    # Create dummy data of shape [Batch, Channels, Time] = [4, 257, 100]
    dummy_input = torch.randn(4, 257, 100)
    print(f"Input shape:  {dummy_input.shape}")

    print("Running forward pass...")
    with torch.no_grad():
        output = model(dummy_input)

    print(f"Output shape: {output.shape}")

    # The typical goal is that output matches input shape for masks
    if output.shape == dummy_input.shape:
        print("Success! The forward pass executed correctly and the output shape matched the input.")
    else:
        print("Warning: Output shape mismatched!")

if __name__ == "__main__":
    main()
