import numpy as np 
from collections import defaultdict
import torch 
from torch import nn
import pytorch3d.transforms as transforms 
from model.resnet import ResNet
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims=(128, 128), activation='tanh', is_dropout=False):
        super().__init__()
        if activation == 'tanh':
            self.activation = torch.tanh
        elif activation == 'relu':
            self.activation = torch.relu
        elif activation == 'sigmoid':
            self.activation = torch.sigmoid

        self.out_dim = hidden_dims[-1]
        self.affine_layers = nn.ModuleList()
        last_dim = input_dim
        for idx, nh in enumerate(hidden_dims):
            self.affine_layers.append(nn.Linear(last_dim, nh))  
            if idx == 0 and is_dropout:
                self.affine_layers.append(nn.Dropout(p=0.5))
            last_dim = nh

    def forward(self, x):
        for affine in self.affine_layers:
            x = self.activation(affine(x))
        return x


def get_sinusoid_encoding_table(n_position, d_hid, padding_idx=None):
    ''' Sinusoid position encoding table '''

    def cal_angle(position, hid_idx):
        return position / np.power(10000, 2 * (hid_idx // 2) / d_hid)

    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(n_position)])

    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    if padding_idx is not None:
        # zero vector for padding dimension
        sinusoid_table[padding_idx] = 0.

    return torch.FloatTensor(sinusoid_table)

def get_subsequent_mask(seq):
    ''' For masking out the subsequent info. '''
    sz_b, len_s = seq.size()
    subsequent_mask = torch.triu(
        torch.ones((len_s, len_s), device=seq.device, dtype=torch.bool), diagonal=1)
    subsequent_mask = subsequent_mask.unsqueeze(0).expand(sz_b, -1, -1) # b x ls x ls
    
    return subsequent_mask


class MultiHeadAttention(nn.Module):
    def __init__(self, n_head, d_model, d_k, d_v):
        super(MultiHeadAttention, self).__init__()

        self.n_head = n_head
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v

        self.w_q = nn.Linear(d_model, n_head*d_k)
        self.w_k = nn.Linear(d_model, n_head*d_k)
        self.w_v = nn.Linear(d_model, n_head*d_v)
        nn.init.normal_(self.w_q.weight, mean=0, std=np.sqrt(2.0/(d_model+d_k)))
        nn.init.normal_(self.w_k.weight, mean=0, std=np.sqrt(2.0/(d_model+d_k)))
        nn.init.normal_(self.w_v.weight, mean=0, std=np.sqrt(2.0/(d_model+d_v)))

        self.temperature = np.power(d_k, 0.5)
        self.attn_dropout = nn.Dropout(0.1)

        self.fc = nn.Linear(n_head*d_v, d_model)
        nn.init.xavier_normal_(self.fc.weight)
        self.layer_norm = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, q, k, v, mask=None):
        # q: BS X T X D, k: BS X T X D, v: BS X T X D, mask: BS X T X T 
        bs, n_q, _ = q.shape
        bs, n_k, _ = k.shape
        bs, n_v, _ = v.shape

        assert n_k == n_v

        residual = q

        q = self.w_q(q).view(bs, n_q, self.n_head, self.d_k).permute(2, 0, 1, 3).contiguous().view(-1, n_q, self.d_k)
        k = self.w_k(k).view(bs, n_k, self.n_head, self.d_k).permute(2, 0, 1, 3).contiguous().view(-1, n_k, self.d_k)
        v = self.w_v(v).view(bs, n_v, self.n_head, self.d_v).permute(2, 0, 1, 3).contiguous().view(-1, n_v, self.d_v)

        attn = torch.bmm(q, k.transpose(1, 2)) # (n_head*bs) X n_q X n_k
        attn = attn / self.temperature

        if mask is not None:
            mask = mask.repeat(self.n_head, 1, 1) # (n_head*bs) x n_q x n_k 
            attn = attn.masked_fill(mask, -np.inf)

        attn = F.softmax(attn, dim=2) # (n_head*bs) X n_q X n_k
        
        attn = self.attn_dropout(attn)
        output = torch.bmm(attn, v) # (n_head*bs) X n_q X d_v

        output = output.view(self.n_head, bs, n_q, self.d_v)
        output = output.permute(1, 2, 0, 3).contiguous().view(bs, n_q, -1)
        # BS X n_q X (n_head*D)

        # output = self.fc(output) # BS X n_q X D
        output = self.dropout(self.fc(output)) # BS X n_q X D
        output = self.layer_norm(output + residual) # BS X n_q X D

        return output, attn


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_in, d_hid):
        super(PositionwiseFeedForward, self).__init__()

        self.w_1 = nn.Conv1d(d_in, d_hid, 1)
        self.w_2 = nn.Conv1d(d_hid, d_in, 1)
        self.layer_norm = nn.LayerNorm(d_in)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        # x: BS X N X D
        residual = x
        output = x.transpose(1, 2) # BS X D X N
        output = self.w_2(F.relu(self.w_1(output))) # BS X D X N
        output = output.transpose(1, 2) # BS X N X D
        output = self.dropout(output)
        output = self.layer_norm(output + residual) # BS X N X D

        return output


class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_head, d_k, d_v):
        super(DecoderLayer, self).__init__()

        self.self_attn = MultiHeadAttention(n_head, d_model, d_k, d_v)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_model)

    def forward(self, decoder_input, self_attn_time_mask, self_attn_padding_mask):
        # decode_input: BS X T X D
        # time_mask: BS X T X T (padding postion are ones)
        # padding_mask: BS X T (padding position are zeros, diff usage from above)
        bs, dec_len, dec_hidden = decoder_input.shape
        
        decoder_out, dec_self_attn = self.self_attn(decoder_input, decoder_input, decoder_input, \
                                mask=self_attn_time_mask)
        # BS X T X D, BS X T X T
        decoder_out *= self_attn_padding_mask.unsqueeze(-1).float()
        # BS X T X D

        decoder_out = self.pos_ffn(decoder_out) # BS X T X D
        decoder_out *= self_attn_padding_mask.unsqueeze(-1).float()

        return decoder_out, dec_self_attn
        # BS X T X D, BS X T X T


class CrossDecoderLayer(nn.Module):
    def __init__(self, d_model, n_head, d_k, d_v):
        super(CrossDecoderLayer, self).__init__()

        self.self_attn = MultiHeadAttention(n_head, d_model, d_k, d_v)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_model)

    def forward(self, decoder_input, k_input, self_attn_time_mask, self_attn_padding_mask):
        # decode_input: BS X T X D
        # k_input: BS X K X D 
        # time_mask: BS X T X T (padding postion are ones)
        # padding_mask: BS X T (padding position are zeros, diff usage from above)
        bs, dec_len, dec_hidden = decoder_input.shape
        
        decoder_out, dec_self_attn = self.self_attn(decoder_input, k_input, k_input, \
                                mask=self_attn_time_mask)
        # BS X T X D, BS X T X T
        decoder_out *= self_attn_padding_mask.unsqueeze(-1).float()
        # BS X T X D

        decoder_out = self.pos_ffn(decoder_out) # BS X T X D
        decoder_out *= self_attn_padding_mask.unsqueeze(-1).float()

        return decoder_out, dec_self_attn
        # BS X T X D, BS X T X T


class Decoder(nn.Module):
    def __init__(
            self,
            d_feats, d_model,
            n_layers, n_head, d_k, d_v, max_timesteps, use_full_attention=False):
        super(Decoder, self).__init__()

        self.start_conv = nn.Conv1d(d_feats, d_model, 1) # (input: 17*3)
        self.position_vec = nn.Embedding.from_pretrained(
            get_sinusoid_encoding_table(max_timesteps+1, d_model, padding_idx=0),
            freeze=True)
        self.layer_stack = nn.ModuleList([DecoderLayer(d_model, n_head, d_k, d_v)
            for _ in range(n_layers)])

        self.use_full_attention = use_full_attention 

    def forward(self, decoder_input, padding_mask, decoder_pos_vec, obj_embedding=None):
        # decoder_input: BS X D X T 
        # padding_mask: BS X 1 X T
        # decoder_pos_vec: BS X 1 X T
        # obj_embedding: BS X 1 X D

        dec_self_attn_list = []

        padding_mask = padding_mask.squeeze(1) # BS X T
        decoder_pos_vec = decoder_pos_vec.squeeze(1) # BS X T

        input_embedding = self.start_conv(decoder_input)  # BS X D X T
        input_embedding = input_embedding.transpose(1, 2) # BS X T X D
        if obj_embedding is not None:
            new_input_embedding = torch.cat((obj_embedding, input_embedding), dim=1) # BS X (T+1) X D 
        else:
            new_input_embedding = input_embedding

        # self.position_vec = self.position_vec.cuda()
        pos_embedding = self.position_vec(decoder_pos_vec) # BS X T X D
        
        # Time mask is same for all blocks, while padding mask differ according to the position of block
        if self.use_full_attention:
            time_mask = None
        else:
            time_mask = get_subsequent_mask(decoder_pos_vec) 
        # BS X T X T (Prev steps are 0, later 1)
       
        dec_output = new_input_embedding + pos_embedding # BS X T X D
        for dec_layer in self.layer_stack:
            dec_output, dec_self_attn = dec_layer(
                dec_output, # BS X T X D
                self_attn_time_mask=time_mask, # BS X T X T
                self_attn_padding_mask=padding_mask) # BS X T

            dec_self_attn_list += [dec_self_attn]

        return dec_output, dec_self_attn_list
        # BS X T X D, list


def upsample_batch(images, scale_factor=4, interpolation=cv2.INTER_CUBIC):
    images = images.cpu().numpy()
    batch_size, channels, height, width = images.shape
    upscaled_height, upscaled_width = height * scale_factor, width * scale_factor
    
    # Initialize an empty array for the upscaled images
    upsampled_images = np.zeros((batch_size, channels, upscaled_height, upscaled_width), dtype=images.dtype)
    
    # Upsample each image in the batch
    for i in range(batch_size):
        for c in range(channels):
            upsampled_images[i, c] = cv2.resize(images[i, c], (upscaled_width, upscaled_height), interpolation=interpolation)
    
    return torch.from_numpy(upsampled_images).cuda()

def get_heading_q_batch(_q):
    q = _q.clone()
    q[:, 1] = 0.0
    q[:, 2] = 0.0
    
    q_norm = torch.norm(q, dim = 1,  p=2).view(-1, 1)

    return q/q_norm

def de_heading_batch(q):
    q_deheaded = get_heading_q_batch(q)
    q_deheaded_inv = transforms.quaternion_invert(q_deheaded)

    return transforms.quaternion_multiply(q_deheaded_inv, q)

def position_loss(gt_pos, pred_pos):
    # gt_pos/pred_pos: (BS*T) X 3 
    return (gt_pos - pred_pos).pow(2).sum(dim=1)

def orientation_loss(gt_quat, pred_quat):
    # gt_quat/pred_quat: (BS*T) X 4
    pred_quat_invert = transforms.quaternion_invert(pred_quat) # (BS*T) X 4 
    dist = transforms.quaternion_multiply(gt_quat, pred_quat_invert) 
    """make the diff quat to be identity"""
    quat_iden = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=gt_quat.dtype, device=gt_quat.device).repeat(gt_quat.size()[0], 1)
    loss = torch.abs(dist) - quat_iden

    return loss.pow(2).sum(dim=1)

class HeadFormer(nn.Module):

    def __init__(self, opt, device):
        super(HeadFormer, self).__init__()
       
        self.opt = opt 

        self.device = device 

        self.htype = htype = "relu"
        self.mlp_hsize = mlp_hsize = [1024, 512, 256]
        self.cnn_fdim = cnn_fdim = 512

        self.transformer_window_size = opt.window 
    
        self.input_of_feats = opt.input_of_feats
        if not self.input_of_feats:
            self.cnn = ResNet(cnn_fdim, running_stats=False, pretrained=True)
            self.freeze_of_cnn = opt.freeze_of_cnn
            if opt.freeze_of_cnn:
                # If freeze the CNN params 
                for param in self.cnn.parameters():
                    param.requires_grad = False

        # New version 
        self.action_n_dec_layers = opt.n_dec_layers
        self.action_n_head = opt.n_head
        self.action_d_k = opt.d_k
        self.action_d_v = opt.d_v 
        self.action_d_feats = self.cnn_fdim # Optical flow feature vector
        self.action_d_model = opt.d_model
        self.action_max_timesteps = opt.window

        # Introduce transformer for capturing sequential information
        # Input: BS X D X T 
        # Output: BS X T X D'
        self.action_transformer = Decoder(d_feats=self.action_d_feats, d_model=self.action_d_model, \
            n_layers=self.action_n_dec_layers, n_head=self.action_n_head, \
            d_k=self.action_d_k, d_v=self.action_d_v, max_timesteps=self.action_max_timesteps, \
            use_full_attention=True) 

        self.action_va_mlp = MLP(self.action_d_model, mlp_hsize, htype)
        self.action_va_fc = nn.Linear(mlp_hsize[-1], 3)

        self.action_dist_mlp = MLP(self.action_d_model, mlp_hsize, htype)
        self.action_dist_fc = nn.Linear(mlp_hsize[-1], 1)

    def va2rot(self, curr_rot, pred_head_vels, dt=1/30):
        '''
        gt_head_pose: B X 4 (head quaternion 4)
        pred_head_vels: B X T X 3 (head angular velocity 3)
        '''
        timesteps = pred_head_vels.shape[1]
       
        # curr_heading = get_heading_q_batch(curr_rot)

        head_rot_seq = [curr_rot] # First frame's head rotation in a sequence. 
        
        for t_idx in range(timesteps):        
            root_qvel = pred_head_vels[:, t_idx, :].float() # B X 3
            
            angv = transforms.quaternion_apply(curr_rot.detach().float(), root_qvel)
            new_rot = transforms.quaternion_multiply(transforms.axis_angle_to_quaternion(angv * dt), curr_rot.detach())
            curr_rot = new_rot/torch.norm(new_rot, dim = 1).reshape(-1, 1)

            head_rot_seq.append(curr_rot)

        head_rot_seq = torch.stack(head_rot_seq, dim=1) # B X T X 4 

        return head_rot_seq

    def prep_padding_mask(self, seq_len):
        # Generate padding mask 
        actual_seq_len = seq_len # BS
        tmp_mask = torch.arange(self.transformer_window_size).expand(seq_len.shape[0], \
        self.transformer_window_size).to(seq_len.device) < actual_seq_len[:, None].repeat(1, self.transformer_window_size)
        # BS X max_timesteps
        padding_mask = tmp_mask[:, None, :]

        return padding_mask
    
    def forward(self, data):
        if self.input_of_feats:
            input_features = data['of'].to(self.device).float() # B X T X D (extract optical features off line)
        else:
            of_data = data['of'].to(self.device) # B X T X 224 X 224 X 2 
            batch_size, seq_len, _, _, _ = of_data.shape
            of_data = torch.cat((of_data, torch.zeros(of_data.shape[:-1] + (1,), device=of_data.device)), dim=-1)
            h, w = 224, 224 
            c = 3
            of_data = of_data.reshape(-1, h, w, c).permute(0, 3, 1, 2) # B X T X 2 X 224 X 224 
            input_features = self.cnn(of_data).reshape(batch_size, seq_len, self.cnn_fdim) # B X T X D 

            if self.freeze_of_cnn:
                input_features = input_features.detach() 

        batch_size, seq_len, _ = input_features.shape 

        decoder_input = input_features.transpose(1, 2) # BS X D X T 

        actual_seq_len = data['seq_len'].to(self.device).float()
        padding_mask = self.prep_padding_mask(actual_seq_len) # BS 

        # Get position vec for position-wise embedding
        bs = decoder_input.shape[0]
        num_steps = decoder_input.shape[2]
        pos_vec = torch.arange(num_steps)+1 # timesteps
        pos_vec = pos_vec[None, None, :].to(self.device).repeat(bs, 1, 1) # BS X 1 X timesteps

        seq_rnn_out, _ = self.action_transformer(decoder_input, padding_mask, pos_vec) # B X T X D

        # Predict angular velocity to integrate to rotation. 
        seq_va_out = self.action_va_mlp(seq_rnn_out)
        seq_va_out = self.action_va_fc(seq_va_out) # B X T X 3

        # Predict distance scalar values used for correcting DROID-SLAM scale. 
        seq_dist_scalar = self.action_dist_mlp(seq_rnn_out)
        seq_dist_scalar = self.action_dist_fc(seq_dist_scalar) # BS X T X 1 
    
        feature_pred = defaultdict(list)  
        feature_pred['head_va'] = seq_va_out # B X T X 3 

        seq_head_rot_quat = self.va2rot(data['head_pose'][:, 0, 3:].to(self.device), seq_va_out) # B X T X 4 
      
        feature_pred['head_rot_quat'] = seq_head_rot_quat # B X (T+1) X 4

        feature_pred['head_dist_scalar'] = seq_dist_scalar # B X T X 1 

        return feature_pred

    def cal_scale_for_slam_w_pred_scale(self, slam_trans, dist_scalar):
        # slam_trans: (T+1) X 3 
        # dist_scalar: T 

        window_size = slam_trans.shape[0] - 1
        # window_size = 10 

        slam_abs_len_list = []
        for t_idx in range(window_size):
            curr_slam_trans = slam_trans[t_idx+1] - slam_trans[t_idx]
            curr_slam_abs_len = torch.linalg.norm(curr_slam_trans)
            
            slam_abs_len_list.append(curr_slam_abs_len)

        if len(slam_abs_len_list) < dist_scalar.shape[0]:
            dist_scalar = dist_scalar[:len(slam_abs_len_list)]
        elif len(slam_abs_len_list) > dist_scalar.shape[0]:
            slam_abs_len_list = slam_abs_len_list[:dist_scalar.shape[0]]
        
        model_abs_len = dist_scalar.mean()
        slam_abs_len = torch.stack(slam_abs_len_list).mean() 

        scale = model_abs_len/slam_abs_len 

        rescaled_trans = [slam_trans[0]]
        for t_idx in range(slam_trans.shape[0]-1):
            curr_slam_trans = slam_trans[t_idx+1] - slam_trans[t_idx]

            rescaled_trans.append(rescaled_trans[-1]+scale*curr_slam_trans)

        rescaled_trans = torch.stack(rescaled_trans)

        return rescaled_trans, scale  

    def forward_for_eval(self, data):
        if self.input_of_feats:
            cnn = ResNet(512, running_stats=False, pretrained=True).cuda()
            of_data = data['image'].to(self.device)
            B, T, C, H, W = of_data.shape
            of_data = of_data.reshape(-1, C, H, W)
            of_data = upsample_batch(of_data)
            input_features = cnn(of_data).reshape(B, T, 512)
            # input_features = data['of'].to(self.device).float() # B X T X D (extract optical features off line)
        else:
            of_data = data['image'].to(self.device) # B X T X 224 X 224 X 2 
            # batch_size, seq_len, _, _, _ = of_data.shape
            # of_data = torch.cat((of_data, torch.zeros(of_data.shape[:-1] + (1,), device=of_data.device)), dim=-1)
            # h, w = 224, 224 
            # c = 3
            # of_data = of_data.reshape(-1, h, w, c).permute(0, 3, 1, 2) # B X T X 2 X 224 X 224 
            B, T, C, H, W = of_data.shape
            of_data = of_data.reshape(-1, C, H, W)
            of_data = upsample_batch(of_data)
            input_features = self.cnn(of_data).reshape(B, T, self.cnn_fdim) # B X T X D 

        timesteps = input_features.shape[1]

        aligned_slam_trans = data['aligned_slam_trans'].to(self.device) # B(1) X T X 3
        aligned_slam_rot_quat = data['aligned_slam_rot_quat'].to(self.device) # B(1) X T X 4 

        seq_head_rot_quat_list = []
        seq_head_dist_scalar_list = []

        stride = self.transformer_window_size
        num_blocks = timesteps//stride + 1

        for b_idx in range(num_blocks):
            curr_input_features = input_features[:, b_idx*stride:(b_idx+1)*stride]
            # curr_input_features = input_features[:, t_idx:t_idx+stride]
            if curr_input_features.shape[1] > 0:
                decoder_input = curr_input_features.transpose(1, 2) # BS X D X T

                actual_seq_len = torch.tensor(decoder_input.shape[2])[None].to(self.device)
                padding_mask = self.prep_padding_mask(actual_seq_len) # BS 

                # Get position vec for position-wise embedding
                bs = decoder_input.shape[0]

                if actual_seq_len < self.transformer_window_size:
                    # Add padding
                    padding_input = torch.zeros(bs, decoder_input.shape[1], \
                    self.transformer_window_size-actual_seq_len).to(decoder_input.device)

                    decoder_input = torch.cat((decoder_input, padding_input), dim=-1)

                num_steps = decoder_input.shape[2]
                pos_vec = torch.arange(num_steps)+1 # timesteps
                pos_vec = pos_vec[None, None, :].to(self.device).repeat(bs, 1, 1) # BS X 1 X timesteps

                curr_seq_rnn_out, _ = self.action_transformer(decoder_input, padding_mask, pos_vec) # B X T X D

                curr_seq_rnn_out = curr_seq_rnn_out[:, :actual_seq_len, :] 

                # Predict angular velocity to integrate to rotation. 
                seq_va_out = self.action_va_mlp(curr_seq_rnn_out)
                seq_va_out = self.action_va_fc(seq_va_out) # B X T X 3

                # Predict distance scalar values used for correcting DROID-SLAM scale. 
                seq_dist_scalar = self.action_dist_mlp(curr_seq_rnn_out)
                seq_dist_scalar = self.action_dist_fc(seq_dist_scalar) # BS X T X 1 

                seq_head_dist_scalar_list.append(seq_dist_scalar)

                if b_idx == 0:
                    curr_seq_head_rot_quat = self.va2rot(data['head_pose'][:, 0, 3:].to(self.device), seq_va_out.detach())
                    seq_head_rot_quat_list.append(curr_seq_head_rot_quat)
                else:
                    curr_seq_head_rot_quat = self.va2rot(prev_head_rot_quat, seq_va_out.detach())
                    # prev_head_pose: B X 4 
                    seq_head_rot_quat_list.append(curr_seq_head_rot_quat[:, 1:, :]) # The first pose is alread in the sequence. 
                
                prev_head_rot_quat = curr_seq_head_rot_quat[:, -1, :] # B X 4

        seq_head_rot_quat = torch.cat(seq_head_rot_quat_list, dim=1) # B(1) X T X 4 
        seq_head_dist_scalar = torch.cat(seq_head_dist_scalar_list, dim=1) # B(1) X T X 1

        seq_head_dist_scalar = seq_head_dist_scalar / self.opt.dist_scale 

        rescaled_trans, pred_scale = self.cal_scale_for_slam_w_pred_scale(aligned_slam_trans[0], seq_head_dist_scalar[0].squeeze(-1))
        # T X 3 

        if rescaled_trans.shape[0] != seq_head_rot_quat.shape[1]:
            seq_head_rot_quat = seq_head_rot_quat[:, :rescaled_trans.shape[0], :]

        seq_head_pose = torch.cat((rescaled_trans[None], seq_head_rot_quat), dim=-1) # B(1) X T X 7 

        feature_pred = defaultdict(list)  

        # assert seq_head_pose.shape[1] == data['head_pose'].shape[1] 
        if seq_head_pose.shape[1] != data['head_pose'].shape[1] and seq_head_pose.shape[1] != data['head_pose'].shape[1]-1:
            print("gt and pred head pose shape different!")
            print("pred shape:{0}".format(seq_head_pose.shape))
            print("gt shape:{0}".format(data['head_pose'].shape))
       
        feature_pred['head_pose'] = seq_head_pose
        feature_pred['pred_scale'] = pred_scale 

        return feature_pred

    def compute_loss(self, feature_pred, data):
        b_size, seq_len, nq = feature_pred['head_va'].shape
    
        pred_head_vels = feature_pred['head_va'].reshape(b_size * seq_len, -1) # (B*T) X D 
        gt_head_vels = data['head_vels'][:, :, 3:].reshape(b_size * seq_len, -1).float().to(self.device) # (B*T) X 3 

        pred_head_pose = feature_pred['head_rot_quat'][:, 1:, :].reshape(b_size * seq_len, -1) # (B*T) X 4
        gt_head_pose = data['head_pose'][:, 1:, 3:].reshape(b_size * seq_len, -1).float().to(self.device) 

        pred_dist_scalar = feature_pred['head_dist_scalar'].reshape(b_size * seq_len, -1) # (B*T) X 1
        gt_dist_scalar = self.get_dist_scalar(data['head_pose'][:, :, :3]).reshape(b_size * seq_len, -1).float().to(pred_dist_scalar.device) # (B*T) X 1  
        gt_dist_scalar = self.opt.dist_scale *  gt_dist_scalar # (B*T) X 1 
        # dist_scale can make the targer larger, maybe better for learning when actual target value is too small 

        va_loss = position_loss(gt_head_vels, pred_head_vels).mean() 
        orient_loss = orientation_loss(gt_head_pose, pred_head_pose).mean()
        dist_loss = (pred_dist_scalar-gt_dist_scalar).pow(2).sum(dim=1).mean()

        loss = self.opt.w_rotation * orient_loss + self.opt.w_va * va_loss + self.opt.w_dist * dist_loss 

        return loss, orient_loss, va_loss, dist_loss 

    def get_dist_scalar(self, gt_head_trans):
        # B X T X 3 
        bs, timesteps, _ = gt_head_trans.shape

        abs_len_list = []
        for t_idx in range(timesteps-1):
            curr_trans = gt_head_trans[:, t_idx+1] - gt_head_trans[:, t_idx] # B X 3 
            curr_abs_len = torch.linalg.vector_norm(curr_trans, dim=1) # B 
            
            abs_len_list.append(curr_abs_len)

        abs_len_list = torch.stack(abs_len_list) # T X B 
        abs_len_list = abs_len_list.transpose(0, 1) # B X T
        return abs_len_list 
