import torch
from torch import nn
import math 
from tqdm.auto import tqdm
from einops import reduce
from inspect import isfunction
import torch.nn.functional as F
import pytorch3d.transforms as transforms
from dataset.data_utils.multi_modal_dataset import quat_ik_torch 
from utils.utils_lafan import rotate_at_frame_smplh
def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def linear_beta_schedule(timesteps):
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype = torch.float64)

def cosine_beta_schedule(timesteps, s = 0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype = torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        denoise_fn,
        condition_module,
        timesteps = 1000,
        loss_type = 'l1',
        objective = 'pred_x0',
        beta_schedule = 'cosine',
        p2_loss_weight_gamma = 0., # p2 loss weight, from https://arxiv.org/abs/2204.00227 - 0 is equivalent to weight of 1 across time - 1. is recommended
        p2_loss_weight_k = 1,
        window = 120,
        device = 'cuda',
    ):
        super().__init__()
        self.denoise_fn = denoise_fn
        self.device = device
        self.condition_module = condition_module
        self.headnet = None
        self.guidance = None
        self.objective = objective

        self.seq_len = window

        if beta_schedule == 'linear':
            betas = linear_beta_schedule(timesteps)
        elif beta_schedule == 'cosine':
            betas = cosine_beta_schedule(timesteps)
        else:
            raise ValueError(f'unknown beta schedule {beta_schedule}')

        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value = 1.)

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.loss_type = loss_type
        self.mse_loss_fn = nn.MSELoss(reduction='sum')

        # helper function to register buffer from float64 to float32
        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))

        register_buffer('betas', betas)
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        register_buffer('posterior_variance', posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min =1e-20)))
        register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

        # calculate p2 reweighting
        register_buffer('p2_loss_weight', (p2_loss_weight_k + alphas_cumprod / (1 - alphas_cumprod)) ** -p2_loss_weight_gamma)

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, t, images, audio, clip_denoised, padding_mask=None):
        condition, _, _ = self.condition_module(images, audio)
        model_output = self.denoise_fn(x, t, y=condition)
        # model_output = self.denoise_fn(x, condition, t)

        if self.objective == 'pred_noise':
            x_start = self.predict_start_from_noise(x, t=t, noise=model_output)
        elif self.objective == 'pred_x0':
            x_start = model_output
        else:
            raise ValueError(f'unknown objective {self.objective}')

        if clip_denoised:
            x_start.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_start, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, x, t, images, audio, clip_denoised=True, padding_mask=None):
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance(x=x, t=t, images=images, audio=audio, \
            clip_denoised=clip_denoised, padding_mask=padding_mask)
        noise = torch.randn_like(x)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @torch.no_grad()
    def p_sample_loop(self, shape, x_start, images, audio, padding_mask=None):
        device = self.betas.device

        b = shape[0]
        x = torch.randn(shape, device=device)

        for i in tqdm(reversed(range(0, self.num_timesteps)), desc='sampling loop time step', total=self.num_timesteps):
            x = self.p_sample(x, torch.full((b,), i, device=device, dtype=torch.long), images, audio, padding_mask=padding_mask)     

        return x # BS X (T,J) X D

    @torch.no_grad()
    def p_sample_loop_sliding_window(self, shape, x_start, cond_mask):
        device = self.betas.device

        b = shape[0]
        assert b == 1
        
        x_all = torch.randn(shape, device=device)
        x_cond_all = x_start * (1. - cond_mask) + \
            cond_mask * torch.randn_like(x_start).to(x_start.device)

        x_blocks = []
        x_cond_blocks = []
        # Divide to blocks to form a batch, then just need run model once to get all the results. 
        num_steps = x_start.shape[1]
        stride = self.window // 2
        for t_idx in range(0, num_steps, stride):
            x = x_all[0, t_idx:t_idx+self.window]
            x_cond = x_cond_all[0, t_idx:t_idx+self.window]

            x_blocks.append(x) # T X D 
            x_cond.append(x_cond) # T X D 

        last_window_x = None 
        last_window_cond = None 
        if x_blocks[-1].shape[0] != x_blocks[0].shape[0]:
            last_window_x = x_blocks[-1][None] # 1 X T X D 
            last_window_cond = x_cond_blocks[-1][None] 

            x_blocks = torch.stack(x_blocks[:-1]) # K X T X D 
            x_cond_blocks = torch.stack(x_cond_blocks[:-1]) # K X T X D 
        else:
            x_blocks = torch.stack(x_blocks) # K X T X D 
            x_cond_blocks = torch.stack(x_cond_blocks) # K X T X D 
       
        for i in tqdm(reversed(range(0, self.num_timesteps)), desc='sampling loop time step', total=self.num_timesteps):
            x_blocks = self.p_sample(x_blocks, torch.full((b,), i, device=device, dtype=torch.long), x_cond_blocks)    

        if last_window_x is not None:
            for i in tqdm(reversed(range(0, self.num_timesteps)), desc='sampling loop time step', total=self.num_timesteps):
                last_window_x = self.p_sample(last_window_x, torch.full((b,), i, device=device, dtype=torch.long), last_window_cond)    

        # Convert from K X T X D to a single sequence.
        seq_res = None  
        # for t_idx in range(0, num_steps, stride):
        num_windows = x_blocks.shape[0]
        for w_idx in range(num_windows):
            if w_idx == 0:
                seq_res = x_blocks[w_idx] # T X D 
            else:
                seq_res = torch.cat((seq_res, x_blocks[self.window-stride:]), dim=0)

        if last_window_x is not None:
            seq_res = torch.cat((seq_res, last_window_x[self.window-stride:]), dim=0)

        return seq_res # BS X T X D

    @torch.no_grad()
    def p_sample_loop_sliding_window_w_canonical(self, ds, shape, global_head_jpos, global_head_jquat, image, audio, cond_mask):
        # shape: BS X T X D 
        # global_head_jpos: BS X T X 3 
        # global_head_jquat: BS X T X 4 
        # cond_mask: BS X T X D 

        device = self.betas.device

        b = shape[0]
        # assert b == 1
        
        x_all = torch.randn(shape, device=device)

        whole_seq_aa_rep = None 
        whole_seq_root_pos = None 
        whole_seq_head_pos = None 

        # Divide to blocks to form a batch, then just need run model once to get all the results. 
        num_steps = global_head_jpos.shape[1]
        # stride = self.seq_len // 2
        overlap_frame_num = 10
        stride = self.seq_len - overlap_frame_num 
        for t_idx in range(0, num_steps, stride):
            curr_x = x_all[:, t_idx:t_idx+self.seq_len]

            if curr_x.shape[1] <= self.seq_len - stride:
                break 

            # Canonicalize current window 
            curr_global_head_quat = global_head_jquat[:, t_idx:t_idx+self.seq_len] # BS X T X 4
            curr_global_head_jpos = global_head_jpos[:, t_idx:t_idx+self.seq_len] # BS X T X 3 

            aligned_head_trans, aligned_head_quat, recover_rot_quat = \
            rotate_at_frame_smplh(curr_global_head_jpos.data.cpu().numpy(), \
            curr_global_head_quat.data.cpu().numpy(), cano_t_idx=0)
            # BS X T' X 3, BS X T' X 4, BS X 1 X 1 X 4  

            aligned_head_trans = torch.from_numpy(aligned_head_trans).to(global_head_jpos.device)
            aligned_head_quat = torch.from_numpy(aligned_head_quat).to(global_head_jpos.device)

            move_to_zero_trans = aligned_head_trans[:, 0:1, :].clone() # Move the head joint x, y to 0,  BS X 1 X 3
            move_to_zero_trans[:, :, 2] = 0 

            aligned_head_trans = aligned_head_trans - move_to_zero_trans # BS X T X 3 

            aligned_head_rot_mat = transforms.quaternion_to_matrix(aligned_head_quat) # BS X T X 3 X 3 
            aligned_head_rot_6d = transforms.matrix_to_rotation_6d(aligned_head_rot_mat) # BS X T X 6  

            head_idx = 15 
            curr_x_start = torch.zeros(aligned_head_rot_6d.shape[0], \
            aligned_head_rot_6d.shape[1], 24*3+24*6).to(aligned_head_rot_6d.device)
            curr_x_start[:, :, head_idx*3:head_idx*3+3] = aligned_head_trans # BS X T X 3
            curr_x_start[:, :, 24*3+head_idx*6:24*3+head_idx*6+6] = aligned_head_rot_6d # BS X T X 6 

            # Normalize data to [-1, 1]
            normalized_jpos = curr_x_start[:, :, :24*3].reshape(-1, 24, 3)
            curr_x_start[:, :, :24*3] = normalized_jpos.reshape(b, -1, 24*3) # BS X T X (22*3)

            curr_cond_mask = cond_mask[:, t_idx:t_idx+self.seq_len] # BS X T X D 
            curr_x_cond = curr_x_start * (1. - curr_cond_mask) + \
            curr_cond_mask * torch.randn_like(curr_x_start).to(curr_x_start.device)
       
            for i in tqdm(reversed(range(0, self.num_timesteps)), desc='sampling loop time step', total=self.num_timesteps):
                curr_x = self.p_sample(curr_x, torch.full((b,), i, device=device, dtype=torch.long), image, audio)    
                # Apply previous window prediction as additional condition, direcly replacement. 
                if t_idx > 0:
                    curr_x[:, :self.seq_len-stride, 24*3:] = prev_res_rot_6d.reshape(b, -1, 24*6)
                    curr_x[:, :self.seq_len-stride, :24*3] = prev_res_jpos.reshape(b, -1, 24*3)

            curr_seq_local_aa_rep, curr_seq_root_pos, curr_seq_head_pos = \
            self.convert_model_res_to_data(ds, curr_x, \
            recover_rot_quat, curr_global_head_jpos) 
            
            if t_idx == 0:
                whole_seq_aa_rep = curr_seq_local_aa_rep # BS X T X 22 X 3
                whole_seq_root_pos = curr_seq_root_pos # BS X T X 3 
                whole_seq_head_pos = curr_seq_head_pos # BS X T X 3 
            else:
                prev_last_pos = whole_seq_head_pos[:, -1:, :].clone() # BS X 1 X 3 
                curr_first_pos = curr_seq_head_pos[:, self.seq_len-stride-1:self.seq_len-stride, :].clone() # BS X 1 X 3
                
                move_trans = prev_last_pos - curr_first_pos # BS X 1 X 3 
                curr_seq_root_pos += move_trans # BS X T X 3 
                curr_seq_head_pos += move_trans 
                
                whole_seq_aa_rep = torch.cat((whole_seq_aa_rep, \
                curr_seq_local_aa_rep[:, self.seq_len-stride:]), dim=1)
                whole_seq_root_pos = torch.cat((whole_seq_root_pos, \
                curr_seq_root_pos[:, self.seq_len-stride:]), dim=1)
                whole_seq_head_pos = torch.cat((whole_seq_head_pos, \
                curr_seq_head_pos[:, self.seq_len-stride:]), dim=1)

            # Convert results to normalized representation for sampling in next window
            tmp_global_rot_quat, tmp_global_jpos = ds.fk_smpl(curr_seq_root_pos.reshape(-1, 3), \
            curr_seq_local_aa_rep.reshape(-1, 24, 3)) 
            # (BS*T) X 22 X 4, (BS*T) X 22 X 3 
            tmp_global_rot_quat = tmp_global_rot_quat.reshape(b, -1, 24, 4)
            tmp_global_jpos = tmp_global_jpos.reshape(b, -1, 24, 3)

            tmp_global_rot_quat = tmp_global_rot_quat[:, -self.seq_len+stride:].clone()
            tmp_global_jpos = tmp_global_jpos[:, -self.seq_len+stride:].clone()
            
            tmp_global_head_quat = tmp_global_rot_quat[:, :, 15, :] # BS X T X 4 
            tmp_global_head_jpos = tmp_global_jpos[:, :, 15, :] # BS X T X 3 

            tmp_aligned_head_trans, tmp_aligned_head_quat, tmp_recover_rot_quat = \
            rotate_at_frame_smplh(tmp_global_head_jpos.data.cpu().numpy(), \
            tmp_global_head_quat.data.cpu().numpy(), cano_t_idx=0)
            # BS X T' X 3, BS X T' X 4, BS X 1 X 1 X 4  

            tmp_aligned_head_trans = torch.from_numpy(tmp_aligned_head_trans).to(tmp_global_head_jpos.device)

            tmp_move_to_zero_trans = tmp_aligned_head_trans[:, 0:1, :].clone() 
            # Move the head joint x, y to 0,  BS X 1 X 3
            tmp_move_to_zero_trans[:, :, 2] *= 0 # 1 X 1 X 3 

            tmp_aligned_head_trans = tmp_aligned_head_trans - tmp_move_to_zero_trans # BS X T X 3 

            tmp_recover_rot_quat = torch.from_numpy(tmp_recover_rot_quat).float().to(tmp_global_rot_quat.device)

            tmp_global_jpos = transforms.quaternion_apply(transforms.quaternion_invert(\
            tmp_recover_rot_quat).repeat(1, tmp_global_jpos.shape[1], \
            tmp_global_jpos.shape[2], 1), tmp_global_jpos) # BS X T X 22 X 3

            tmp_global_jpos -= tmp_move_to_zero_trans[:, :, None, :] 

            prev_res_jpos = tmp_global_jpos.clone() 
            # prev_res_jpos = ds.normalize_jpos_min_max(prev_res_jpos.reshape(-1, 22, 3)).reshape(b, -1, 24, 3) # BS X T X 22 X 3 

            prev_res_global_rot_quat = transforms.quaternion_multiply(transforms.quaternion_invert(\
            tmp_recover_rot_quat).repeat(1, tmp_global_rot_quat.shape[1], \
            tmp_global_rot_quat.shape[2], 1), \
            tmp_global_rot_quat) # BS X T X 22 X 4
            prev_res_rot_mat = transforms.quaternion_to_matrix(prev_res_global_rot_quat) # BS X T X 22 X 3 X 3 
            prev_res_rot_6d = transforms.matrix_to_rotation_6d(prev_res_rot_mat) # BS X T X 22 X 6 

        return whole_seq_aa_rep, whole_seq_root_pos
        # T X 22 X 3, T X 3 

    def convert_model_res_to_data(self, ds, all_res_list, recover_rot_quat, curr_global_head_jpos):
        # all_res_list: BS X T X D 
        # recover_rot_quat: BS X 1 X 1 X 4 
        # curr_global_head_jpos: BS X T X 3 

        # De-normalize jpos 
        use_global_head_pos_for_root_trans = False 

        bs = all_res_list.shape[0]
        global_jpos = all_res_list[:, :, :24*3].reshape(bs, -1, 24, 3) # BS X T X 22 X 3 
      
        # global_jpos = ds.de_normalize_jpos_min_max(normalized_global_jpos.reshape(-1, 22, 3)) # (BS*T) X 22 X 3
        # global_jpos = global_jpos.reshape(bs, -1, 24, 3) # BS X T X 22 X 3 

        global_rot_6d = all_res_list[:, :, 24*3:] # BS X T X (22*6)
        
        bs, num_steps, _, _ = global_jpos.shape
        global_rot_6d = global_rot_6d.reshape(bs, num_steps, 24, 6) # BS X T X 22 X 6 
        
        global_root_jpos = global_jpos[:, :, 0, :] # BS X T X 3 

        head_idx = 15 
        global_head_jpos = global_jpos[:, :, head_idx, :] # BS X T X 3 

        global_rot_mat = transforms.rotation_6d_to_matrix(global_rot_6d) # BS X T X 22 X 3 X 3
        global_quat = transforms.matrix_to_quaternion(global_rot_mat) # BS X T X 22 X 4 
        recover_rot_quat = torch.from_numpy(recover_rot_quat).to(global_quat.device) # BS X 1 X 1 X 4 
        ori_global_quat = transforms.quaternion_multiply(recover_rot_quat, global_quat) # BS X T X 22 X 4 
        ori_global_root_jpos = global_root_jpos # BS X T X 3 
        ori_global_root_jpos = transforms.quaternion_apply(recover_rot_quat.squeeze(1).repeat(1, num_steps, 1), \
                        ori_global_root_jpos) # BS X T X 3 

        ori_global_head_jpos = transforms.quaternion_apply(recover_rot_quat.squeeze(1).repeat(1, num_steps, 1), \
                        global_head_jpos) # BS X T X 3 

        # Convert global join rotation to local joint rotation
        ori_global_rot_mat = transforms.quaternion_to_matrix(ori_global_quat) # BS X T X 22 X 3 X 3
        ori_local_rot_mat = quat_ik_torch(ori_global_rot_mat.reshape(-1, 24, 3, 3), parents=ds.parents).reshape(bs, -1, 24, 3, 3) # BS X T X 22 X 3 X 3 
        ori_local_aa_rep = transforms.matrix_to_axis_angle(ori_local_rot_mat) # BS X T X 22 X 3 

        if use_global_head_pos_for_root_trans: 
            zero_root_trans = torch.zeros(ori_local_aa_rep.shape[0], ori_local_aa_rep.shape[1], 3).to(ori_local_aa_rep.device).float()
            betas = torch.zeros(bs, 10).to(zero_root_trans.device).float()
            gender = ["male"] * bs 

            _, mesh_jnts = ds.fk_smpl(zero_root_trans.reshape(-1, 3), ori_local_aa_rep.reshape(-1, 24, 3))
            # (BS*T) X 22 X 4, (BS*T) X 22 X 3 
            mesh_jnts = mesh_jnts.reshape(bs, -1, 24, 3) # BS X T X 22 X 3 

            head_idx = 15 
            wo_root_trans_head_pos = mesh_jnts[:, :, head_idx, :] # BS X T X 3 

            calculated_root_trans = ori_global_head_jpos - wo_root_trans_head_pos # BS X T X 3 

            return ori_local_aa_rep, calculated_root_trans, ori_global_head_jpos

        return ori_local_aa_rep, ori_global_root_jpos, ori_global_head_jpos

    @torch.no_grad()
    def sample(self, x_start, images, audio, padding_mask=None):
        # naive conditional sampling by replacing the noisy prediction with input target data. 
        _B, _, _, _D = x_start.shape
        x_start = x_start.reshape(_B, -1, _D)
        # _B, _, _D = x_start.shape
        self.denoise_fn.eval()
        self.condition_module.eval()
        sample_res = self.p_sample_loop(x_start.shape, \
                x_start, images, audio)
        # BS X (T,J) X D
        # sample_res = sample_res.reshape(_B, self.seq_len, -1) # BS X T X D
        self.denoise_fn.train()
        self.condition_module.train()
        return sample_res  

    @torch.no_grad()
    def sample_sliding_window(self, x_start, cond_mask):
        # If the sequence is longer than trained max window, divide 
        self.denoise_fn.eval()
        sample_res = self.p_sample_loop_sliding_window(x_start.shape, \
                x_start, cond_mask)
        # BS X T X D
        self.denoise_fn.train()
        return sample_res  

    @torch.no_grad()
    def sample_sliding_window_w_canonical(self, ds, global_head_jpos, global_head_jquat, x_start, image, audio, cond_mask):
        # If the sequence is longer than trained max window, divide 
        self.denoise_fn.eval()
        sample_res = self.p_sample_loop_sliding_window_w_canonical(ds, x_start.shape, \
                global_head_jpos, global_head_jquat, image, audio, cond_mask)
        # BS X T X D
        self.denoise_fn.train()
        return sample_res  

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    @property
    def loss_fn(self):
        if self.loss_type == 'l1':
            return F.l1_loss
        elif self.loss_type == 'l2':
            return F.mse_loss
        else:
            raise ValueError(f'invalid loss type {self.loss_type}')

    def p_losses(self, x_start, images, audio, t, noise=None, padding_mask=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        x = self.q_sample(x_start=x_start, t=t, noise=noise) # noisy motion in noise level t. 

        condition, _, _ = self.condition_module(images, audio)
        model_out = self.denoise_fn(x, t, y=condition)
        # model_out = self.denoise_fn(x, condition, t)

        if self.objective == 'pred_noise':
            target = noise
        elif self.objective == 'pred_x0':
            target = x_start
        else:
            raise ValueError(f'unknown objective {self.objective}')

        # Predicting both head pose and other joints' pose. 
        if padding_mask is not None:
            loss = self.loss_fn(model_out, target, reduction = 'none') * padding_mask[:, 0, 1:][:, :, None]
        else:
            loss = self.loss_fn(model_out, target, reduction = 'none') # BS X T X D 
           
        loss = reduce(loss, 'b ... -> b (...)', 'mean')

        loss = loss * extract(self.p2_loss_weight, t, loss.shape)
        diffusion_loss = loss.mean()
        # align_loss = self.mse_loss_fn(attn_out_audio_weights, attn_out_image_weights)
        # tot_loss = diffusion_loss + align_loss
        return diffusion_loss

    def forward(self, x_start, images, audio, padding_mask=None):
        # x_start: BS X T X J x 9 
        # image : BS X T x 3 X H X W
        # audio : BS X T X F
        # padding_mask: BS X 1 X T 
        _B, _T, _J, _D = x_start.shape
        # _B, _T, _D = x_start.shape
        t = torch.randint(0, self.num_timesteps, (_B,), device=x_start.device).long()
        # print("t:{0}".format(t))
        x_start = x_start.reshape(_B, -1, _D)
        curr_loss = self.p_losses(x_start, images, audio, t, padding_mask=padding_mask)

        return curr_loss
    
    def set_guidance(self, guidance):
        self.guidance = guidance
    
    def set_headnet(self, headnet):
        self.headnet = headnet
        

if __name__ == '__main__':
    device = 'cuda'
    window = 120
    diffusion_model = GaussianDiffusion(input_dim=3, embed_dim=256, depth=8, window=window, device=device).to(device)
    x_start = torch.randn(10, window, 24, 3).to(device)
    image = torch.randn(10, window, 3, 64, 64).to(device)
    audio = torch.randn(10, window, 35).to(device)
    loss = diffusion_model(x_start, image, audio)
    param = sum(p.numel() for p in diffusion_model.parameters() if p.requires_grad)
    print(param)
    print(loss)