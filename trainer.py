import os
import torch
from torch.optim import Adam
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import pytorch3d.transforms as transforms 
from ema_pytorch import EMA
from dataset.data_utils.multi_modal_dataset import MultiModalDataset, quat_ik_torch, run_smpl_model
from utils.vis.blender_vis_mesh_motion import run_blender_rendering_and_save2video, save_verts_faces_to_mesh_file
import logging
def cycle(dl):
    while True:
        for data in dl:
            yield data

class Trainer(object):
    def __init__(
        self,
        opt,
        diffusion_model,
        device,
        *,
        ema_decay = 0.995,
        train_batch_size = 32,
        train_lr = 1e-4,
        train_num_steps = 10000000,
        gradient_accumulate_every = 2,
        amp = False,
        step_start_ema = 2000,
        ema_update_every = 10,
        save_and_sample_every = 200000,
        results_folder = './results',
        run_demo=False,
    ):
        super().__init__()

        self.tb_writer = SummaryWriter(opt.exp_dir + '/tensorboard')
        self.device = device
        self.model = diffusion_model
        self.ema = EMA(diffusion_model, beta=ema_decay, update_every=ema_update_every)
        self.step_start_ema = step_start_ema
        self.save_and_sample_every = save_and_sample_every
        self.batch_size = train_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every
        self.train_num_steps = train_num_steps
        self.optimizer = Adam(diffusion_model.parameters(), lr=train_lr)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, train_num_steps)
        self.step = 0
        self.amp = amp
        self.scaler = GradScaler(enabled=amp)
        
        self.results_folder = results_folder

        self.vis_folder = results_folder.replace("weights", "vis_res")

        self.opt = opt 

        if run_demo:
            self.ds = MultiModalDataset(self.opt, train=False, window=opt.window, run_demo=True) 
        else:
            self.prep_dataloader(window_size=opt.window)

        self.window = opt.window 

        self.bm_dict = self.ds.bm_dict
        self.parents = self.ds.parents

    def prep_dataloader(self, window_size):
        # Define dataset
        train_dataset = MultiModalDataset(self.opt, mode='train', window=window_size, device=self.device)
        val_dataset = MultiModalDataset(self.opt, mode = 'val', window=window_size, device=self.device)

        self.ds = train_dataset 
        self.val_ds = val_dataset
        self.dl = cycle(DataLoader(self.ds, batch_size=self.batch_size, shuffle=True, pin_memory=True, num_workers=8))
        self.val_dl = cycle(DataLoader(self.val_ds, batch_size=2, shuffle=False, pin_memory=True, num_workers=2))

    def save(self, milestone):
        data = {
            'step': self.step,
            'model': self.model.state_dict(),
            'ema': self.ema.state_dict(),
            'scaler': self.scaler.state_dict()
        }
        torch.save(data, os.path.join(self.results_folder, 'model-'+str(milestone)+'.pt'))

    def load(self, milestone):
        data = torch.load(os.path.join(self.results_folder, 'model-'+str(milestone)+'.pt'))

        self.step = data['step']
        self.model.load_state_dict(data['model'], strict=False)
        self.ema.load_state_dict(data['ema'], strict=False)
        self.scaler.load_state_dict(data['scaler'])

    def load_weight_path(self, weight_path):
        data = torch.load(weight_path)

        self.step = data['step']
        self.model.load_state_dict(data['model'], strict=False)
        self.ema.load_state_dict(data['ema'], strict=False)
        # self.scaler.load_state_dict(data['scaler'])

    def train(self):
        init_step = self.step 
        loss = 0.
        for idx in range(init_step, self.train_num_steps):
            self.optimizer.zero_grad()
            self.model.train()
            nan_exists = False # If met nan in loss or gradient, need to skip to next data. 
            data_dict = next(self.dl)

            motion = data_dict['motion'].to(self.device)
            images = data_dict['image'].to(self.device)
            audio = data_dict['audio'].to(self.device)
            # padding_mask = self.prep_padding_mask(motion, data_dict['seq_len']).to(self.device)

            with autocast(enabled = self.amp):
                tot_loss = self.model(motion, images, audio, padding_mask=None)
                if torch.isnan(tot_loss).item():
                    logging.info('WARNING: NaN loss. Skipping to next data...')
                    nan_exists = True 
                    torch.cuda.empty_cache()
                    continue

                self.scaler.scale(tot_loss).backward()
                curr_lr = self.scheduler.get_last_lr()[0]
                self.tb_writer.add_scalar('Train/Learing_rate', curr_lr, idx)
                # check gradients
                parameters = [p for p in self.model.parameters() if p.grad is not None]
                total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), 2.0).to(self.device) for p in parameters]), 2.0)
                if torch.isnan(total_norm):
                    logging.info('WARNING: NaN gradients. Skipping to next data...')
                    nan_exists = True 

                    torch.cuda.empty_cache()
                    continue
                if self.amp:
                    current_grad_scale = self.scaler._scale.item() if self.amp else 1.0
                    self.tb_writer.add_scalar('Train/Grad Scale', current_grad_scale, idx)
                self.tb_writer.add_scalar('Train/Total_loss', tot_loss.item(), idx)
                # self.tb_writer.add_scalar('Train/Diffusion_loss', diffusion_loss.item(), idx)
                # self.tb_writer.add_scalar('Train/Align_loss', align_loss.item(), idx)


                if idx % self.opt.log_interval == 0:
                    logging.info(f"Step: {idx} || TotLoss: {tot_loss.item()} || LR: {curr_lr}" )
                        

            if nan_exists:
                continue

            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.ema.update()
            self.scheduler.step()
            
            if self.step != 0 and self.step % self.save_and_sample_every == 0:
                self.ema.ema_model.eval()
                with torch.no_grad():
                    milestone = self.step // self.save_and_sample_every
                    # val_data_dict = next(self.val_dl) 
                    # val_motion = val_data_dict['motion'].to(self.device)
                    # val_image = val_data_dict['image'].to(self.device)
                    # val_audio = val_data_dict['audio'].to(self.device)
                    # padding_mask = self.prep_padding_mask(val_motion, val_data_dict['seq_len']).to(self.device)

                    # all_res_list = self.ema.ema_model.sample(val_motion, val_image, val_audio, padding_mask=padding_mask)
                
                logging.info(f"Saving checkpoint at step {self.step}...")
                self.save(milestone)

                # Visualization
                logging.info(f"Generating visualization at step {self.step}...")
                # bs_for_vis = 2
                # self.gen_vis_res(val_motion[:bs_for_vis], self.step, vis_gt=True)
                # self.gen_vis_res(all_res_list[:bs_for_vis], self.step)

            self.step += 1

        logging.info("Training finished.")

    def prep_head_condition_mask(self, data, joint_idx=15):
        # data: BS X T X D 
        # head_idx = 15 
        # Condition part is zeros, while missing part is ones. 
        mask = torch.ones_like(data).to(data.device)

        cond_pos_dim_idx = joint_idx * 3 
        cond_rot_dim_idx = 24 * 3 + joint_idx * 6
        mask[:, :, cond_pos_dim_idx:cond_pos_dim_idx+3] = torch.zeros(data.shape[0], data.shape[1], 3).to(data.device)
        mask[:, :, cond_rot_dim_idx:cond_rot_dim_idx+6] = torch.zeros(data.shape[0], data.shape[1], 6).to(data.device)

        return mask 

    def prep_padding_mask(self, val_data, seq_len):
        # Generate padding mask 
        actual_seq_len = seq_len + 1 # BS, + 1 since we need additional timestep for noise level 
        tmp_mask = torch.arange(self.window+1).expand(val_data.shape[0], \
        self.window+1) < actual_seq_len[:, None].repeat(1, self.window+1)
        # BS X max_timesteps
        padding_mask = tmp_mask[:, None, :].to(val_data.device)

        return padding_mask 

    def cond_sample_res(self):
        weights = os.listdir(self.results_folder)
        weights_paths = [os.path.join(self.results_folder, weight) for weight in weights]
        weight_path = max(weights_paths, key=os.path.getctime)

        print(f"Loaded weight: {weight_path}")

        milestone = weight_path.split("/")[-1].split("-")[-1].replace(".pt", "")
        
        self.load(milestone)
        self.ema.ema_model.eval()
        num_sample = 4
        with torch.no_grad():
            for s_idx in range(num_sample):
                val_data_dict = next(self.val_dl)
                val_motion = val_data_dict['motion'].cuda()
                val_images = val_data_dict['images'].cuda()
                val_audio = val_data_dict['audio'].cuda()

                padding_mask = self.prep_padding_mask(val_motion, val_data_dict['seq_len'])
                
                all_res_list = self.ema.ema_model.sample(x_start=val_motion,
                                                         images = val_images, audio = val_audio, 
                                                         padding_mask=padding_mask)

                vis_tag = "test_head_cond_sample_"+str(s_idx)

                max_num = 1
                self.gen_vis_res(val_motion[:max_num], vis_tag, vis_gt=True)
                self.gen_vis_res(all_res_list[:max_num], vis_tag)

    def full_body_gen_cond_sliding_window(self, image, audio, head_pose):
        # head_pose: BS X T X 7 
        self.ema.ema_model.eval()

        global_head_jpos = head_pose[:, :, :3] # BS X T X 3 
        global_head_quat = head_pose[:, :, 3:] # BS X T X 4 

        data = torch.zeros(head_pose.shape[0], head_pose.shape[1], 24*3+24*6).to(head_pose.device) # BS X T X D 

        with torch.no_grad():
            cond_mask = self.prep_head_condition_mask(data) # BS X T X D 

            local_aa_rep, seq_root_pos = self.ema.ema_model.sample_sliding_window_w_canonical(self.ds, \
            global_head_jpos, global_head_quat, x_start=data, image=image, audio=audio, cond_mask=cond_mask) 
            # BS X T X 22 X 3, BS X T X 3       

        return local_aa_rep, seq_root_pos # T X 22 X 3, T X 3  

    def gen_vis_res(self, all_res_list, step, vis_gt=False):
        # all_res_list: N X T X D 
        num_seq = all_res_list.shape[0]
        all_res_list = all_res_list.detach().cpu()
        
        # all_res_list = all_res_list.reshape(-1, self.window, 24, 9)
        # all_res_list = all_res_list*self.ds.std + self.ds.mean
        # all_res_list = all_res_list.reshape(-1, self.window, 24*9)
        # num_seq = all_res_list.shape[0]
        # global_jpos = all_res_list[:, :, :24*3].reshape(num_seq, -1, 24, 3)
        # global_root_jpos = global_jpos[:, :, 0, :].clone() # N X T X 3
        # global_rot_6d = all_res_list[:, :, 24*3:].reshape(num_seq, -1, 24, 6)
        # global_rot_mat = transforms.rotation_6d_to_matrix(global_rot_6d) # N X T X 22 X 3 X 3 
        
        all_res_list = all_res_list*self.ds.std + self.ds.mean
        local_rot_6d = all_res_list[:, :, :24*6].reshape(num_seq, -1, 24, 6)
        local_rot_mat = transforms.rotation_6d_to_matrix(local_rot_6d) # N X T X 24 X 3 X 3 
        global_root_jpos = all_res_list[:, :, -3:] # N X T X 3
        for idx in range(num_seq):
            curr_local_rot_mat = local_rot_mat[idx] # T X 24 X 3 X 3 
            # curr_local_rot_mat = quat_ik_torch(curr_global_rot_mat, self.parents) # T X 24 X 3 X 3 
            curr_local_rot_aa_rep = transforms.matrix_to_axis_angle(curr_local_rot_mat) # T X 24 X 3 
            # curr_global_rot_mat = global_rot_mat[idx] # T X 22 X 3 X 3 
            # curr_local_rot_mat = quat_ik_torch(curr_global_rot_mat, self.parents) # T X 22 X 3 X 3 
            # curr_local_rot_aa_rep = transforms.matrix_to_axis_angle(curr_local_rot_mat) # T X 22 X 3 
            
            curr_global_root_jpos = global_root_jpos[idx] # T X 3
            move_xy_trans = curr_global_root_jpos.clone()[0:1] # 1 X 3 
            move_xy_trans[:, 2] = 0 
            root_trans = curr_global_root_jpos - move_xy_trans # T X 3 

            # Generate global joint position 
            bs = 1
            gender = ["male"] * bs 
            
            # print(root_trans.device, curr_local_rot_aa_rep.device, self.bm_dict['male'].device)
            mesh_jnts, mesh_verts, mesh_faces = \
            run_smpl_model(root_trans[None], \
                        curr_local_rot_aa_rep[None], gender, \
                        self.bm_dict)
            # BS(1) X T' X 24 X 3, BS(1) X T' X Nv X 3
            
            dest_mesh_vis_folder = os.path.join(self.vis_folder, "blender_mesh_vis", str(step))
            if not os.path.exists(dest_mesh_vis_folder):
                os.makedirs(dest_mesh_vis_folder)

            if vis_gt:
                mesh_save_folder = os.path.join(dest_mesh_vis_folder, \
                                "objs_step_"+str(step)+"_bs_idx_"+str(idx)+"_gt")
                out_rendered_img_folder = os.path.join(dest_mesh_vis_folder, \
                                "imgs_step_"+str(step)+"_bs_idx_"+str(idx)+"_gt")
                out_vid_file_path = os.path.join(dest_mesh_vis_folder, \
                                "vid_step_"+str(step)+"_bs_idx_"+str(idx)+"_gt.mp4")
            else:
                mesh_save_folder = os.path.join(dest_mesh_vis_folder, \
                                "objs_step_"+str(step)+"_bs_idx_"+str(idx))
                out_rendered_img_folder = os.path.join(dest_mesh_vis_folder, \
                                "imgs_step_"+str(step)+"_bs_idx_"+str(idx))
                out_vid_file_path = os.path.join(dest_mesh_vis_folder, \
                                "vid_step_"+str(step)+"_bs_idx_"+str(idx)+".mp4")

            # Visualize the skeleton 
            if vis_gt:
                dest_skeleton_vis_path = os.path.join(dest_mesh_vis_folder, \
                                "vid_step_"+str(step)+"_bs_idx_"+str(idx)+"_skeleton_gt.gif")
            else:
                dest_skeleton_vis_path = os.path.join(dest_mesh_vis_folder, \
                                "vid_step_"+str(step)+"_bs_idx_"+str(idx)+"_skeleton.gif")
            # show3Dpose_animation_smpl22(channels.data.cpu().numpy(), dest_skeleton_vis_path) 

            # For visualizing human mesh only 
            logging.info("Saving mesh to {}".format(mesh_save_folder))
            save_verts_faces_to_mesh_file(mesh_verts.data.cpu().numpy()[0], mesh_faces.data.cpu().numpy(), mesh_save_folder)
            # run_blender_rendering_and_save2video(mesh_save_folder, out_rendered_img_folder, out_vid_file_path)

    def gen_full_body_vis(self, root_trans, curr_local_rot_aa_rep, dest_mesh_vis_folder, seq_name, vis_gt=False):
        # root_trans: T X 3 
        # curr_local_rot_aa_rep: T X 22 X 3 

        # Generate global joint position 
        bs = 1
        betas = torch.zeros(bs, 16).to(root_trans.device)
        gender = ["male"] * bs 

        mesh_jnts, mesh_verts, mesh_faces = run_smpl_model(root_trans[None].float(), \
        curr_local_rot_aa_rep[None].float(), gender, self.ds.bm_dict)
        # BS(1) X T' X 24 X 3, BS(1) X T' X Nv X 3
    
        if vis_gt:
            mesh_save_folder = os.path.join(dest_mesh_vis_folder, seq_name, \
                            "objs_gt")
            out_rendered_img_folder = os.path.join(dest_mesh_vis_folder, seq_name, \
                            "imgs_gt")
            out_vid_file_path = os.path.join(dest_mesh_vis_folder, \
                            seq_name+"_vid_gt.mp4")
        else:
            mesh_save_folder = os.path.join(dest_mesh_vis_folder, seq_name, \
                            "objs")
            out_rendered_img_folder = os.path.join(dest_mesh_vis_folder, seq_name, \
                            "imgs")
            out_vid_file_path = os.path.join(dest_mesh_vis_folder, \
                            seq_name+"_vid.mp4")

        # For visualizing human mesh only 
        save_verts_faces_to_mesh_file(mesh_verts.data.cpu().numpy()[0], \
        mesh_faces.data.cpu().numpy(), mesh_save_folder)
 #       run_blender_rendering_and_save2video(mesh_save_folder, \
  #      out_rendered_img_folder, out_vid_file_path)

        return mesh_jnts, mesh_verts