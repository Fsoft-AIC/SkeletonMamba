import torch
from torch import nn
from model.head_estimation_transformer import Decoder, PositionwiseFeedForward
from model.resnet import ResNet

class AudioEncoder(nn.Module):
    def __init__(self, in_feats=35, out_feats=213, d_model=128, n_layers=4, n_head=8, d_k=64, d_v=64, max_timesteps=1000):
        super().__init__()
        self.audio_encoder = Decoder(d_feats=in_feats, d_model=d_model, n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v, max_timesteps=max_timesteps)
        self.audio_mlp = nn.Linear(d_model, out_feats)
        self.audio_norm = nn.LayerNorm(out_feats)
    
    def forward(self, audio):
        B, T, D = audio.shape
        padding_mask = torch.ones(B, T).unsqueeze(1).to(audio.device).bool() # B X 1 X T
        posvec = torch.arange(T).unsqueeze(0).unsqueeze(0).repeat(B, 1, 1).to(audio.device)  # B X 1 X T
        audio = audio.transpose(1, 2) # B X D X T
        audio_embed, _ = self.audio_encoder(audio, padding_mask, posvec)
        audio_embed = self.audio_mlp(audio_embed) # B X T X D
        audio_embed = self.audio_norm(audio_embed)
        return audio_embed

class ConditionModule(nn.Module):
    def __init__(self, window, d_feats):
        super().__init__()
        self.optical_flow = ResNet(d_feats, fix_params=True, pretrained=True)
        # self.learnable_token = nn.Parameter(torch.ones(window, d_feats))
        self.audio_encoder = AudioEncoder(out_feats=d_feats)
        self.cross_attention_audio = nn.MultiheadAttention(d_feats, 8, batch_first=True)
        self.cross_attention_image = nn.MultiheadAttention(d_feats, 8, batch_first=True)
        self.fusion_condition = nn.Sequential(
            nn.Linear(2*d_feats, d_feats),
            nn.LayerNorm(d_feats),
            nn.GELU(),
        )
        self.norm1 = nn.LayerNorm(d_feats)
        self.norm2 = nn.LayerNorm(d_feats)
        self.mlp1 = PositionwiseFeedForward(d_feats, d_feats)
        self.mlp2 = PositionwiseFeedForward(d_feats, d_feats)
    def forward(self, images, audio):
        B, T, C, H, W = images.shape
        images = images.reshape(-1,C,H,W)
        image_embed = self.optical_flow(images).reshape(B, T, -1) # B X T X D
        audio_embed = self.audio_encoder(audio) # B X T X D

        # learnale_tokens = self.learnable_token.unsqueeze(0).repeat(B, 1, 1)
        attn_out_audio, attn_out_audio_weights = self.cross_attention_audio(image_embed, audio_embed, audio_embed)
        attn_out_image, attn_out_image_weights = self.cross_attention_image(audio_embed, image_embed, image_embed)
        audio_embed = self.norm1(audio_embed + attn_out_audio)
        audio_embed = self.mlp1(audio_embed)
        image_embed = self.norm2(image_embed + attn_out_image)
        image_embed = self.mlp2(image_embed)
        conditioned = torch.cat((image_embed, audio_embed), dim=-1) # B X T X 2D
        conditioned = self.fusion_condition(conditioned) # B X T X D
        return conditioned, attn_out_audio_weights, attn_out_image_weights