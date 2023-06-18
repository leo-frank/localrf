# Copyright (c) Meta Platforms, Inc. and affiliates.

import math

import torch
from models.tensorBase import AlphaGridMask

from models.tensoRF import TensorVMSplit

from utils.utils import mtx_to_sixD, sixD_to_mtx
from utils.ray_utils import get_ray_directions_lean, get_rays_lean
from utils.utils import N_to_reso

def ids2pixel_view(W, H, ids):
    """
    Regress pixel coordinates from ray indices
    """
    col = ids % W
    row = (ids // W) % H
    view_ids = ids // (W * H)
    return col, row, view_ids

def ids2pixel(W, H, ids):
    """
    Regress pixel coordinates from ray indices
    注意，这里的ids不一定是在n_px_per_frame范围内的，因为我们在sample dataset的时候把idx变换到不同的图像上去，每个图像上都分配256个ray
    所以有的id很大，这里返回的是对应到图像上的col row，
    举个例子800*900的图像，一共有720000个像素，像素索引是[0，719999]，那么对于720000这个索引（超出了第一张图乡）来说，也会映射到0（col）,0(row)
    """
    col = ids % W
    row = (ids // W) % H
    return col, row

class LocalTensorfs(torch.nn.Module):
    """
    Self calibrating local tensorfs.
    """

    def __init__(
        self,
        fov,
        n_init_frames,
        n_overlap,
        WH,
        n_iters_per_frame,
        n_iters_reg,
        lr_R_init,
        lr_t_init,
        lr_i_init,
        lr_exposure_init,
        rf_lr_init,
        rf_lr_basis,
        lr_decay_target_ratio,
        N_voxel_list,
        update_AlphaMask_list,
        camera_prior,
        device,
        lr_upsample_reset,
        **tensorf_args,
    ):

        super(LocalTensorfs, self).__init__()

        self.fov = fov
        self.n_init_frames = n_init_frames
        self.n_overlap = n_overlap
        self.W, self.H = WH
        self.n_iters_per_frame = n_iters_per_frame
        self.n_iters_reg_per_frame = n_iters_reg
        self.lr_R_init, self.lr_t_init, self.lr_i_init, self.lr_exposure_init = lr_R_init, lr_t_init, lr_i_init, lr_exposure_init
        self.rf_lr_init, self.rf_lr_basis, self.lr_decay_target_ratio = rf_lr_init, rf_lr_basis, lr_decay_target_ratio
        self.N_voxel_per_frame_list = N_voxel_list
        self.update_AlphaMask_per_frame_list = update_AlphaMask_list
        self.device = device
        self.camera_prior = camera_prior
        self.tensorf_args = tensorf_args
        self.is_refining = False
        self.lr_upsample_reset = lr_upsample_reset

        self.lr_factor = 1
        self.regularize = True
        self.n_iters_reg = self.n_iters_reg_per_frame
        self.n_iters = self.n_iters_per_frame
        self.update_AlphaMask_list = update_AlphaMask_list
        self.N_voxel_list = N_voxel_list

        # Setup pose and camera parameters
        self.r_c2w, self.t_c2w, self.exposure = torch.nn.ParameterList(), torch.nn.ParameterList(), torch.nn.ParameterList()
        """
        r_optimizers: rotation optimizer;
        t_optimizers: translation optimizer;
        """
        self.r_optimizers, self.t_optimizers, self.exp_optimizers, self.pose_linked_rf = [], [], [], [] 
        # pose_linked_rf: pose_linked_rf[i] stores the radiance field idx of self.r_c2w[i] and self.t_c2w[i]
        self.blending_weights = torch.nn.Parameter(
            torch.ones([1, 1], device=self.device, requires_grad=False), 
            requires_grad=False,
        )
        for _ in range(n_init_frames):
            self.append_frame()

        if self.camera_prior is not None:
            focal = self.camera_prior["transforms"]["fl_x"]
            focal *= self.W / self.camera_prior["transforms"]["w"]
        else:
            fov = fov * math.pi / 180
            focal = self.W / math.tan(fov / 2) / 2
        
        self.init_focal = torch.nn.Parameter(torch.Tensor([focal]).to(self.device))

        self.focal_offset = torch.nn.Parameter(torch.ones(1, device=device))
        self.center_rel = torch.nn.Parameter(0.5 * torch.ones(2, device=device))

        if lr_i_init > 0:
            self.intrinsic_optimizer = torch.optim.Adam([self.focal_offset, self.center_rel], betas=(0.9, 0.99), lr=self.lr_i_init)


        # Setup radiance fields
        self.tensorfs = torch.nn.ParameterList()        # store all local radiance fields
        self.rf_optimizers, self.rf_iter = [], []       # rf_optimizers: all localRF's optimizer; rf_iter: ?
        self.world2rf = torch.nn.ParameterList()        # 这个是干啥的，是不是存放
        self.append_rf()

    def append_rf(self, n_added_frames=1):
        self.is_refining = False
        if len(self.tensorfs) > 0:
            n_overlap = min(n_added_frames, self.n_overlap, self.blending_weights.shape[0] - 1) # 重叠的帧数
            weights_overlap = 1 / n_overlap + torch.arange(
                0, 1, 1 / n_overlap
            ) # 用于表示重叠区域内的权重变化。
            self.blending_weights.requires_grad = False
            self.blending_weights[-n_overlap :, -1] = 1 - weights_overlap # 越靠近localRF中心的frame权重越大
            # new_blending_weights.shape 为 (M, 1)。其中，M 表示行数，表示帧数；N 表示列数，表示权重维度。
            new_blending_weights = torch.zeros_like(self.blending_weights[:, 0:1]) # (M, 1) # new_blending_weights指的是第二个localRF的权重
            new_blending_weights[-n_overlap :, 0] = weights_overlap
            self.blending_weights = torch.nn.Parameter(
                torch.cat([self.blending_weights, new_blending_weights], dim=1),
                requires_grad=False,
            )
            world2rf = -self.t_c2w[-1].clone().detach()         # the location of last estimated camera pose, why minus ? 
        else:   # first localRF
            world2rf = torch.zeros(3, device=self.device)

        self.tensorfs.append(TensorVMSplit(device=self.device, **self.tensorf_args))

        self.world2rf.append(world2rf.clone().detach())
        
        self.rf_iter.append(0)

        grad_vars = self.tensorfs[-1].get_optparam_groups(
            self.rf_lr_init, self.rf_lr_basis
        )
        self.rf_optimizers.append(torch.optim.Adam(grad_vars, betas=(0.9, 0.99)))
   
    def append_frame(self):
        if len(self.r_c2w) == 0: # the first initial frame,
            self.r_c2w.append(torch.eye(3, 2, device=self.device)) # 轴角（axis-angle）表示法。在这种表示法中，旋转矩阵通过一个单位轴向量和一个旋转角度来描述旋转操作。其中，r[..., 0] 是单位轴向量，r[..., 1] 是旋转角度。
            self.t_c2w.append(torch.zeros(3, device=self.device))

            self.pose_linked_rf.append(0)            
        else: # initialize the new pose (index p + 1) using the current frame at the end of the trajectory
            self.r_c2w.append(mtx_to_sixD(sixD_to_mtx(self.r_c2w[-1].clone().detach()[None]))[0]) # TODO: 为什么来来回回转换？
            self.t_c2w.append(self.t_c2w[-1].clone().detach())

            self.blending_weights = torch.nn.Parameter(
                torch.cat([self.blending_weights, self.blending_weights[-1:, :]], dim=0), # (M, N) -> (M+1, N)
                requires_grad=False,
            )

            rf_ind = int(torch.nonzero(self.blending_weights[-1, :])[0]) # 最后一行非零元素的索引 ？？？
            self.pose_linked_rf.append(rf_ind)
                
        self.exposure.append(torch.eye(3, 3, device=self.device))

        if self.camera_prior is not None:
            idx = len(self.r_c2w) - 1
            rel_pose = self.camera_prior["rel_poses"][idx]
            last_r_c2w = sixD_to_mtx(self.r_c2w[-1].clone().detach()[None])[0]
            self.r_c2w[-1] = last_r_c2w @ rel_pose[:3, :3]
            self.t_c2w[-1].data += last_r_c2w @ rel_pose[:3, 3]
            
        self.r_optimizers.append(torch.optim.Adam([self.r_c2w[-1]], betas=(0.9, 0.99), lr=self.lr_R_init)) 
        self.t_optimizers.append(torch.optim.Adam([self.t_c2w[-1]], betas=(0.9, 0.99), lr=self.lr_t_init)) 
        self.exp_optimizers.append(torch.optim.Adam([self.exposure[-1]], betas=(0.9, 0.99), lr=self.lr_exposure_init)) 

    def optimizer_step_poses_only(self, loss):
        """
        update all parameters about pose, i.e. self.r_optimizers, self.t_optimizers
        """
        for idx in range(len(self.r_optimizers)):
            if self.pose_linked_rf[idx] == len(self.rf_iter) - 1 and self.rf_iter[-1] < self.n_iters:
                self.r_optimizers[idx].zero_grad()
                self.t_optimizers[idx].zero_grad()
        
        loss.backward()

        # Optimize poses
        for idx in range(len(self.r_optimizers)):
            if self.pose_linked_rf[idx] == len(self.rf_iter) - 1 and self.rf_iter[-1] < self.n_iters:
                self.r_optimizers[idx].step()
                self.t_optimizers[idx].step()
                
    def optimizer_step(self, loss, optimize_poses):
        """
        self.r_optimizers, self.t_optimizers, self.exp_optimizers
        """
        if self.rf_iter[-1] == 0:
            self.lr_factor = 1
            self.n_iters = self.n_iters_per_frame
            self.n_iters_reg = self.n_iters_reg_per_frame
            

        elif self.rf_iter[-1] == 1:
            n_training_frames = (self.blending_weights[:, -1] > 0).sum()
            self.n_iters = int(self.n_iters_per_frame * n_training_frames)
            self.n_iters_reg = int(self.n_iters_reg_per_frame * n_training_frames)
            self.lr_factor = self.lr_decay_target_ratio ** (1 / self.n_iters)
            self.N_voxel_list = {int(key * n_training_frames): self.N_voxel_per_frame_list[key] for key in self.N_voxel_per_frame_list}
            self.update_AlphaMask_list = [int(update_AlphaMask * n_training_frames) for update_AlphaMask in self.update_AlphaMask_per_frame_list]

        self.regularize = self.rf_iter[-1] < self.n_iters_reg

        for idx in range(len(self.r_optimizers)):
            if self.pose_linked_rf[idx] == len(self.rf_iter) - 1 and self.rf_iter[-1] < self.n_iters:
                # Poses
                if optimize_poses:
                    for param_group in self.r_optimizers[idx].param_groups:
                        param_group["lr"] *= self.lr_factor
                    for param_group in self.t_optimizers[idx].param_groups:
                        param_group["lr"] *= self.lr_factor
                    self.r_optimizers[idx].zero_grad()
                    self.t_optimizers[idx].zero_grad()
                
                # Exposure
                if self.lr_exposure_init > 0:
                    for param_group in self.exp_optimizers[idx].param_groups:
                        param_group["lr"] *= self.lr_factor
                    self.exp_optimizers[idx].zero_grad()

        
        
        # Intrinsics
        if (
            self.lr_i_init > 0 and 
            self.blending_weights.shape[1] == 1 and 
            self.is_refining
        ):
            for param_group in self.intrinsic_optimizer.param_groups:
                param_group["lr"] *= self.lr_factor
            self.intrinsic_optimizer.zero_grad()

        # tensorfs
        for optimizer, iteration in zip(self.rf_optimizers, self.rf_iter):
            if iteration < self.n_iters:
                optimizer.zero_grad()

        loss.backward()

        # Optimize RFs
        self.rf_optimizers[-1].step()
        if self.is_refining:
            for param_group in self.rf_optimizers[-1].param_groups:
                param_group["lr"] = param_group["lr"] * self.lr_factor

        # Increase RF resolution
        if self.rf_iter[-1] in self.N_voxel_list:
            n_voxels = self.N_voxel_list[self.rf_iter[-1]]
            reso_cur = N_to_reso(n_voxels, self.tensorfs[-1].aabb)
            self.tensorfs[-1].upsample_volume_grid(reso_cur)

            if self.lr_upsample_reset:
                print("reset lr to initial")
                grad_vars = self.tensorfs[-1].get_optparam_groups(
                    self.rf_lr_init, self.rf_lr_basis
                )
                self.rf_optimizers[-1] = torch.optim.Adam(grad_vars, betas=(0.9, 0.99))

        # Update alpha mask
        if iteration in self.update_AlphaMask_list:
            reso_mask = (self.tensorfs[-1].gridSize / 2).int()
            self.tensorfs[-1].updateAlphaMask(tuple(reso_mask))

        # Update pose related
        for idx in range(len(self.r_optimizers)):
            if self.pose_linked_rf[idx] == len(self.rf_iter) - 1 and self.rf_iter[-1] < self.n_iters:
                # Optimize poses
                if optimize_poses:
                    self.r_optimizers[idx].step()
                    self.t_optimizers[idx].step()
                # Optimize exposures
                if self.lr_exposure_init > 0:
                    self.exp_optimizers[idx].step()
        
        # Optimize intrinsics 为什么要优化内参？
        if (
            self.lr_i_init > 0 and 
            self.blending_weights.shape[1] == 1 and
            self.is_refining 
        ):
            self.intrinsic_optimizer.step()

        if self.is_refining:
            self.rf_iter[-1] += 1

        can_add_rf = self.rf_iter[-1] >= self.n_iters - 1
        return can_add_rf

    def get_cam2world(self, view_ids=None, starting_id=0):
        """
        return all poses we learned in matrix format
        Outputs:
            (N, 3, 4)
        """
        if view_ids is not None:
            r_c2w = torch.stack([self.r_c2w[view_id] for view_id in view_ids], dim=0)
            t_c2w = torch.stack([self.t_c2w[view_id] for view_id in view_ids], dim=0)
        else:
            r_c2w = torch.stack(list(self.r_c2w[starting_id:]), dim=0)
            t_c2w = torch.stack(list(self.t_c2w[starting_id:]), dim=0)
        return torch.cat([sixD_to_mtx(r_c2w), t_c2w[..., None]], dim = -1)

    def get_kwargs(self):
        kwargs = {
            "camera_prior": None,
            "fov": self.fov,
            "n_init_frames": self.n_init_frames,
            "n_overlap": self.n_overlap,
            "WH": (self.W, self.H),
            "n_iters_per_frame": self.n_iters_per_frame,
            "n_iters_reg": self.n_iters_reg_per_frame,
            "lr_R_init": self.lr_R_init,
            "lr_t_init": self.lr_t_init,
            "lr_i_init": self.lr_i_init,
            "lr_exposure_init": self.lr_exposure_init,
            "rf_lr_init": self.rf_lr_init,
            "rf_lr_basis": self.rf_lr_basis,
            "lr_decay_target_ratio": self.lr_decay_target_ratio,
            "lr_decay_target_ratio": self.lr_decay_target_ratio,
            "N_voxel_list": self.N_voxel_per_frame_list,
            "update_AlphaMask_list": self.update_AlphaMask_per_frame_list,
            "lr_upsample_reset": self.lr_upsample_reset,
        }
        kwargs.update(self.tensorfs[0].get_kwargs())

        return kwargs

    def save(self, path):
        kwargs = self.get_kwargs()
        ckpt = {"kwargs": kwargs, "state_dict": self.state_dict()}
        torch.save(ckpt, path)

    def load(self, state_dict):
        # TODO A bit hacky?
        import re
        n_frames = 0
        for key in state_dict:
            if re.fullmatch(r"r_c2w.[0-9]*", key):
                n_frames += 1
            if re.fullmatch(r"tensorfs.[1-9][0-9]*.density_plane.0", key):
                self.tensorf_args["gridSize"] = [state_dict[key].shape[2], state_dict[key].shape[3], state_dict[f"{key[:-15]}density_line.0"].shape[2]]
                self.append_rf()

        for i in range(len(self.tensorfs)):
            if f"tensorfs.{i}.alphaMask.aabb" in state_dict:
                alpha_volume = state_dict[f'tensorfs.{i}.alphaMask.alpha_volume'].to(self.device)
                aabb = state_dict[f'tensorfs.{i}.alphaMask.aabb'].to(self.device)
                self.tensorfs[i].alphaMask = AlphaGridMask(self.device, aabb, alpha_volume)


        for _ in range(n_frames - len(self.r_c2w)):
            self.append_frame()
        
        self.blending_weights = torch.nn.Parameter(
            torch.ones_like(state_dict["blending_weights"]), requires_grad=False
        )

        self.load_state_dict(state_dict)

    def get_dist_to_last_rf(self):
        return torch.norm(self.t_c2w[-1] + self.world2rf[-1])

    def get_reg_loss(self, tvreg, TV_weight_density, TV_weight_app, L1_weight_inital):
        tv_loss = 0
        l1_loss = 0
        if self.rf_iter[-1] < self.n_iters:
            if TV_weight_density > 0:
                tv_weight = TV_weight_density * (self.lr_factor ** self.rf_iter[-1])
                tv_loss += self.tensorfs[-1].TV_loss_density(tvreg).mean() * tv_weight
                
            if TV_weight_app > 0:
                tv_weight = TV_weight_app * (self.lr_factor ** self.rf_iter[-1])
                tv_loss += self.tensorfs[-1].TV_loss_app(tvreg).mean() * tv_weight
    
            if L1_weight_inital > 0:
                l1_loss += self.tensorfs[-1].density_L1() * L1_weight_inital
        return tv_loss, l1_loss

    def focal(self, W):
        return self.init_focal * self.focal_offset * W / self.W 
    def center(self, W, H): # 这个函数就是返回（W/2, H/2）
        return torch.Tensor([W, H]).to(self.center_rel) * self.center_rel

    def forward(
        self,
        ray_ids,
        view_ids,
        W,
        H,
        white_bg=True,
        is_train=True,
        cam2world=None,
        world2rf=None,
        blending_weights=None,
        chunk=16384,
        test_id=False,
        floater_thresh=0,
    ):
        i, j = ids2pixel(W, H, ray_ids)
        directions = get_ray_directions_lean(i, j, self.focal(W), self.center(W, H)) # 得到所有在各自坐标轴下的direction（相机坐标系），还没有变换到世界坐标系

        if blending_weights is None:
            blending_weights = self.blending_weights[view_ids].clone() # (n_views, N), 一行的每一个元素代表了某个localrf对这个view的weights
        if cam2world is None:
            cam2world = self.get_cam2world(view_ids)    # 得到n_views个图像的位姿（当然，我们要训练这些个位姿） (N, 3, 4)
        if world2rf is None:
            world2rf = self.world2rf

        # Train a single RF at a time
        if is_train:
            blending_weights[:, -1] = 1             # 只用最后一个localRF对n_views进行训练，其他的localRF不参与训练？？奇怪，这好像不符合原文的blending定义
            blending_weights[:, :-1] = 0

        if is_train:
            active_rf_ids = [len(self.tensorfs) - 1]    # 只把最后一个localRF看做是active的
        else:
            active_rf_ids = torch.nonzero(torch.sum(blending_weights, dim=0))[:, 0].tolist() # 求出所有参与训练的localRF
        ij = torch.stack([i, j], dim=-1) # (N, 2)
        if len(active_rf_ids) == 0:
            print("****** No valid RF")
            return torch.ones([ray_ids.shape[0], 3]), torch.ones_like(ray_ids).float(), torch.ones_like(ray_ids).float(), directions, ij

        cam2rfs = {} # 对于每个localRF，其看到的n_views个图像位姿是不一样的，中间有一些平移上的差别
        for rf_id in active_rf_ids:
            cam2rf = cam2world.clone()
            cam2rf[:, :3, 3] += world2rf[rf_id] # world2rf存放着rf_id与对应center的对应关系，是一个负数

            cam2rfs[rf_id] = cam2rf

        for key in cam2rfs:
            cam2rfs[key] = cam2rfs[key].repeat_interleave(ray_ids.shape[0] // view_ids.shape[0], dim=0)
        blending_weights_expanded = blending_weights.repeat_interleave(ray_ids.shape[0] // view_ids.shape[0], dim=0)
        rgbs = torch.zeros_like(directions)  # (B, 3)
        depth_maps = torch.zeros_like(directions[..., 0])  # (B, 1)
        N_rays_all = ray_ids.shape[0] # (B)
        chunk = chunk // len(active_rf_ids) # 似乎Batch=4096要比chunk还要小？？我不确定
        for chunk_idx in range(N_rays_all // chunk + int(N_rays_all % chunk > 0)):
            if chunk_idx != 0:
                torch.cuda.empty_cache()
            directions_chunk = directions[chunk_idx * chunk : (chunk_idx + 1) * chunk]
            blending_weights_chunk = blending_weights_expanded[
                chunk_idx * chunk : (chunk_idx + 1) * chunk
            ]

            for rf_id in active_rf_ids:
                # 每个localRF都要去训练输入的ray
                blending_weight_chunk = blending_weights_chunk[:, rf_id]
                cam2rf = cam2rfs[rf_id][chunk_idx * chunk : (chunk_idx + 1) * chunk]

                rays_o, rays_d = get_rays_lean(directions_chunk, cam2rf)
                rays = torch.cat([rays_o, rays_d], -1).view(-1, 6)

                rgb_map_t, depth_map_t = self.tensorfs[rf_id](
                    rays,
                    is_train=is_train,
                    white_bg=white_bg,
                    N_samples=-1,
                    refine=self.is_refining,
                    floater_thresh=floater_thresh,
                )

                rgbs[chunk_idx * chunk : (chunk_idx + 1) * chunk] = (
                    rgbs[chunk_idx * chunk : (chunk_idx + 1) * chunk] + 
                    rgb_map_t * blending_weight_chunk[..., None] # 注意这里，是有混合过程的
                )
                depth_maps[chunk_idx * chunk : (chunk_idx + 1) * chunk] = (
                    depth_maps[chunk_idx * chunk : (chunk_idx + 1) * chunk] + 
                    depth_map_t * blending_weight_chunk # 注意这里，是有混合过程的
                )

        if self.lr_exposure_init > 0:  # 曝光补偿，曝光补偿的实现好像就是很简单，只是乘上了一个系数？
            # TODO: cleanup
            if test_id:
                view_ids_m = torch.maximum(view_ids - 1, torch.tensor(0, device=view_ids.device))
                view_ids_m[view_ids_m==view_ids] = 1
                
                view_ids_p = torch.minimum(view_ids + 1, torch.tensor(len(self.exposure) - 1, device=view_ids.device))
                view_ids_p[view_ids_m==view_ids] = len(self.exposure) - 2
                
                exposure_stacked = torch.stack(list(self.exposure), dim=0).clone().detach()
                exposure = (exposure_stacked[view_ids_m] + exposure_stacked[view_ids_p]) / 2  
            else:
                exposure = torch.stack(list(self.exposure), dim=0)[view_ids]
                
            exposure = exposure.repeat_interleave(ray_ids.shape[0] // view_ids.shape[0], dim=0)
            rgbs = torch.bmm(exposure, rgbs[..., None])[..., 0]
        rgbs = rgbs.clamp(0, 1) # 我不确定这会带来多大的影响。毕竟之前没有用过clamp

        return rgbs, depth_maps, directions, ij