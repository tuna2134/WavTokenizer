from encoder.utils import convert_audio
import torchaudio
import torch
from decoder.pretrained import WavTokenizer

device = torch.device('cpu')

config_path = "./configs/wavtokenizer_smalldata_frame75_3s_nq1_code4096_dim512_kmeans200_attn.yaml"
model_path = "./WavTokenizer_small_320_24k_4096.ckpt"
audio_path = "./tada.wav"

wavtokenizer = WavTokenizer.from_pretrained0802(config_path, model_path)
wavtokenizer = wavtokenizer.to(device)


wav, sr = torchaudio.load(audio_path)
wav = convert_audio(wav, sr, 24000, 1) 
bandwidth_id = torch.tensor([0])
wav = wav.to(device)
features, discrete_code = wavtokenizer.encode_infer(wav, bandwidth_id=bandwidth_id)


def forward(self, features, bandwidth_id):
    return self.decode(features, bandwidth_id=bandwidth_id)

wavtokenizer.forward = forward.__get__(wavtokenizer)

torch.onnx.register_custom_op_symbolic("aten::lift_fresh", lambda g, x: x, 13)
torch.onnx.export(
    wavtokenizer,
    (features, bandwidth_id),
    "decoder.onnx",
    input_names=["features", "bandwidth_id"],
    verbose=True,
    opset=22
)