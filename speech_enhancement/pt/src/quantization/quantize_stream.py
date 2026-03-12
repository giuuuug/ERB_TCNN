from pathlib import Path
import onnx
import numpy as np
from onnxruntime.quantization import quantize_static, CalibrationMethod, QuantFormat, QuantType, CalibrationDataReader
from onnxruntime.quantization.shape_inference import quant_pre_process
from onnxruntime.tools.onnx_model_utils import fix_output_shapes, make_dim_param_fixed
from common.quantization import define_extra_options
from onnxruntime import InferenceSession


class StreamDataReader(CalibrationDataReader):
    '''
    DataReader for the calibration during onnx quantization for streaming models.
    Properly loops over time and maintains state between frames.
    '''
    def __init__(self, quant_dl, session):
        self.quant_dl = quant_dl
        self.enum_dataloader = None
        self.session = session
        self.input_names = [i.name for i in session.get_inputs()]
        
    def get_next(self):
        if self.enum_dataloader is None:
            self.enum_dataloader = iter(self.quant_dl)
            self.current_batch = None
            self.t = 0
            self.states = {name: np.zeros(shape, dtype=np.float32) for name, shape in zip(self.input_names[1:], [i.shape for i in self.session.get_inputs()][1:])}
            
        if self.current_batch is None or self.t >= self.current_batch.shape[-1]:
            try:
                batch = next(self.enum_dataloader)
                if isinstance(batch, (tuple, list)):
                    self.current_batch = np.abs(batch[0].numpy()) # Magnitude spec
                else:
                    self.current_batch = np.abs(batch.numpy())
                self.t = 0
            except StopIteration:
                return None
                
        input_frame = self.current_batch[:, :, self.t:self.t+1].astype(np.float32)
        
        inputs_dict = {self.input_names[0]: input_frame, **self.states}
        
        # Advance the states to get realistic dynamic range distribution
        outs = self.session.run(None, inputs_dict)
        for i, name in enumerate(self.input_names[1:]):
             self.states[name] = outs[i+1]
             
        self.t += 1
        return inputs_dict

    def rewind(self):
        self.enum_dataloader = None


class SEONNXPTQQuantizer_Stream:
    '''Post-training quantizer for ONNX speech enhancement models.'''
    def __init__(self, cfg, model, dataloaders):
        self.cfg = cfg
        self.model = model
        self.quant_dl = dataloaders["quant_dl"]

        self.op_types_to_quantize = cfg.quantization.onnx_quant_parameters.op_types_to_quantize
        assert (isinstance(self.op_types_to_quantize, list) or self.op_types_to_quantize is None), "op_types_to_quantize must be a list of str or None"
        
        self.calibrate_method = getattr(CalibrationMethod, cfg.quantization.onnx_quant_parameters.calibrate_method)
        self.extra_options = define_extra_options(cfg=cfg)
        self.output_dir = Path(cfg.output_dir, cfg.general.saved_models_dir)
        self.output_dir.mkdir(exist_ok=True)

        self.float_onnx_model = onnx.load(self.model._model_path)

        self.data_reader = StreamDataReader(quant_dl=self.quant_dl, session=self.model)
        
    def quantize(self):
        onnx_prep_path = Path(self.output_dir, "preprocessed_model.onnx") 
        
        quant_pre_process(input_model=self.model._model_path, output_model_path=onnx_prep_path)
        print(f"[INFO] Saved preprocessed float ONNX model at {onnx_prep_path}")

        onnx_prep_model = onnx.load(onnx_prep_path)

        orig_opsets = self.float_onnx_model.opset_import
        del onnx_prep_model.opset_import[:]
        for op in orig_opsets:
            opset = onnx_prep_model.opset_import.add()
            opset.domain = op.domain
            opset.version = op.version

        print("Opset imports after cleanup :")
        for opset in onnx_prep_model.opset_import:
            print("opset domain=%r version=%r" % (opset.domain, opset.version))

        onnx.save(onnx_prep_model, onnx_prep_path)

        quantized_model_path = Path(self.output_dir, "quantized_model_int8.onnx")
        per_channel = self.cfg.quantization.granularity == 'per_channel'
        quantize_static(onnx_prep_path,
                        quantized_model_path,
                        self.data_reader,
                        op_types_to_quantize=self.op_types_to_quantize,
                        per_channel=per_channel,
                        reduce_range=self.cfg.quantization.onnx_quant_parameters.reduce_range,
                        weight_type=QuantType.QInt8,
                        activation_type=QuantType.QInt8,
                        calibrate_method=self.calibrate_method,
                        extra_options=self.extra_options)
        
        quantized_static_model_path = Path(self.output_dir, "quantized_model_int8_static.onnx")
        quant_model = onnx.load(quantized_model_path)

        make_dim_param_fixed(quant_model.graph,
                             param_name=self.cfg.quantization.static_axis_name,
                             value=self.cfg.quantization.static_sequence_length)
        fix_output_shapes(quant_model)
        onnx.save(quant_model, quantized_static_model_path)

        print("[INFO] Successfully converted quantized model to static input shape")

        print("\n [INFO] Quantization complete\n"
              f"Quantized model with dynamic input shape saved at {quantized_model_path}\n"
              f"Quantized model with static input shape saved at {quantized_static_model_path}")

        quantized_model_session = InferenceSession(quantized_model_path)
        quantized_static_model_session = InferenceSession(quantized_static_model_path)
        return quantized_model_session, quantized_static_model_session

