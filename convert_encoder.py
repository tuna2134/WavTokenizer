from encoder.utils import convert_audio
import torchaudio
import torch
from decoder.pretrained import WavTokenizer


device = torch.device("cpu")

config_path = "./configs/wavtokenizer_smalldata_frame75_3s_nq1_code4096_dim512_kmeans200_attn.yaml"
model_path = "./WavTokenizer_small_320_24k_4096.ckpt"
audio_path = "./tada.wav"

wavtokenizer = WavTokenizer.from_pretrained0802(config_path, model_path)
wavtokenizer = wavtokenizer.to(device)


def forward(self, wav, sr, bandwidth_id):
    wav = convert_audio(wav, sr, 24000, 1)
    return self.encode_infer(wav, bandwidth_id=bandwidth_id)


wavtokenizer.forward = forward.__get__(wavtokenizer)

wav, sr = torchaudio.load(audio_path)
bandwidth_id = torch.tensor([0])
wav = wav.to(device)


torch.onnx.export(
    wavtokenizer,
    (wav, sr, bandwidth_id),
    "encoder.onnx",
    input_names=["wav", "bandwidth_id"],
)
