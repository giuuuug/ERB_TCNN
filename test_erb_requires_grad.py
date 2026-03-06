import torch
import sys
import os

# Adapt path to ERB_TCNN model
sys.path.append(os.path.join(os.path.dirname(__file__), "speech_enhancement/pt/src/models"))

try:
    from erb_tcnn import ERBTCNN
except ImportError as e:
    print(f"Error importing ERBTCNN: {e}")
    sys.exit(1)

def main():
    print("Instantiating ERBTCNN model...")
    model = ERBTCNN(in_channels=257)
    
    # Check requires_grad for self.erb
    print("Checking if `self.erb` parameters have requires_grad=False...")
    
    if hasattr(model, 'erb'):
        all_false = True
        for name, param in model.erb.named_parameters():
            print(f"  {name}: requires_grad={param.requires_grad}")
            if param.requires_grad:
                all_false = False
                
        if len(list(model.erb.parameters())) == 0:
            print("  self.erb has no parameters (might be parameterless operations).")
        elif all_false:
            print("Success! All parameters in `self.erb` have requires_grad=False.")
        else:
            print("Warning: Some parameters in `self.erb` have requires_grad=True.")
    else:
        print("Error: `self.erb` not found in the model.")

if __name__ == "__main__":
    main()
