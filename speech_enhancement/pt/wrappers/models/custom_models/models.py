from common.registries.model_registry import MODEL_WRAPPER_REGISTRY
import speech_enhancement.pt.src.models
import torch
from common.utils import log_to_file

# Two separate functions here in case we need to separate the model loading logic later on
@MODEL_WRAPPER_REGISTRY.register(framework="torch",model_name="erbtcnn", use_case='speech_enhancement')
def get_erbtcnn(cfg):
    return _get_model(cfg)

@MODEL_WRAPPER_REGISTRY.register(framework="torch",model_name="stfttcnn", use_case='speech_enhancement')
def get_stfttcnn(cfg):
    return _get_model(cfg)

@MODEL_WRAPPER_REGISTRY.register(framework="torch",model_name="convlstm", use_case='speech_enhancement')
def get_convlstm(cfg):
    return _get_model(cfg)

def _get_model(cfg):
    # Note : model is sent to the appropriate device in the trainer/evaluator class and not here
    # Keep compatibility with the old config files by still accepting model.model_type
    model_type = cfg.model.model_name if cfg.model.model_name else cfg.model.model_type
    model_specific_args = cfg.model_specific
    model = getattr(speech_enhancement.pt.src.models, model_type)(**model_specific_args)

    log_to_file(cfg.output_dir, f"Model type: {model_type}")
    # If a state dict is given in cfg, load it
    if cfg.model.state_dict_path:
        state_dict = torch.load(cfg.model.state_dict_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict)
        print(f"Loaded state dict at {cfg.model.state_dict_path}")
        log_to_file(cfg.output_dir, f"Loaded model state dict at: {cfg.model.state_dict_path}")

    return model


PT_MODEL_FNS = {
    'erbtcnn': get_erbtcnn,
    'stfttcnn': get_stfttcnn,
    'convlstm': get_convlstm
}

for model_name, wrapper_fn in PT_MODEL_FNS.items():
    # Add _pt suffix to fn name 
    wrapper_fn.__name__ = f"{model_name}_pt"
    # Expose the wrapper in the module's global namespace
    globals()[wrapper_fn.__name__] = wrapper_fn