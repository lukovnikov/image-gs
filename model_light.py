import logging
import math
import os
import sys
from time import perf_counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from fused_ssim import fused_ssim
from lpips import LPIPS
from pytorch_msssim import MS_SSIM
from torchvision.transforms.functional import gaussian_blur, to_pil_image

from gsplat import (
    project_gaussians_2d_scale_rot,
    rasterize_gaussians_no_tiles,
    rasterize_gaussians_sum,
)
from utils.flip import LDRFLIPLoss
from utils.image_utils import (
    compute_image_gradients,
    get_grid,
    get_psnr,
    load_images,
    save_image,
    separate_image_channels,
    visualize_added_gaussians,
    visualize_gaussians,
)
from utils.misc_utils import clean_dir, get_latest_ckpt_step, save_cfg, set_random_seed
from utils.quantization_utils import ste_quantize
from utils.saliency_utils import get_smap


class GSImage(torch.nn.Module):
    def __init__(self, num_gaussians, feat_dim, size, dtype, device):
        super(GSImage, self).__init__()
        self.feat_dim = feat_dim
        self.dtype = dtype
        self.device = device
        self.img_h, self.img_w = size
        self.register_buffer("size", torch.LongTensor(size))
        self.xy = nn.Parameter(torch.rand(num_gaussians, 2, dtype=self.dtype, device=self.device), requires_grad=True)
        self.scale = nn.Parameter(torch.ones(num_gaussians, 2, dtype=self.dtype, device=self.device), requires_grad=True)
        self.rot = nn.Parameter(torch.zeros(num_gaussians, 1, dtype=self.dtype, device=self.device), requires_grad=True)
        self.feat = nn.Parameter(torch.rand(num_gaussians, self.feat_dim, dtype=self.dtype, device=self.device), requires_grad=True)

    @classmethod
    def from_statedict(cls, statedict):
        num_gaussians = statedict['xy'].shape[0]
        feat_dim = statedict['feat'].shape[1]
        size = (statedict['size'][0].item(), statedict['size'][1].item())
        dtype = statedict['xy'].dtype
        device = statedict['xy'].device
        gsimage = cls(num_gaussians=num_gaussians, feat_dim=feat_dim, size=size, dtype=dtype, device=device)
        gsimage.load_state_dict(statedict)
        return gsimage


class GaussianSplatting2D(nn.Module):
    def __init__(self, args):
        super(GaussianSplatting2D, self).__init__()
        self.evaluate = args.eval
        set_random_seed(seed=args.seed)
        self.dtype = torch.float32
        self._init_target(args)
        self._init_bit_precision(args)
        self._init_gaussians(args)
        self._init_loss(args)
        self._init_optimization(args)

        self.min_steps = args.min_steps
        self.eval_steps = args.eval_steps
        
        self.init_mode = args.init_mode
        self.init_random_ratio = args.init_random_ratio
        self.smap_filter_size = args.smap_filter_size

    def _init_target(self, args):
        self.gamma = args.gamma
        self.downsample = args.downsample
        if self.downsample:
            self.downsample_ratio = float(args.downsample_ratio)
        self.block_h, self.block_w = 16, 16  # Warning: Must match hardcoded value in CUDA kernel, modify with caution
        if self.downsample:
            self.gt_images_upsampled = self.gt_images
            self.img_h_upsampled, self.img_w_upsampled = self.img_h, self.img_w
            self.tile_bounds_upsampled = self.tile_bounds

    def _init_bit_precision(self, args):
        self.quantize = args.quantize
        self.pos_bits = args.pos_bits
        self.scale_bits = args.scale_bits
        self.rot_bits = args.rot_bits
        self.feat_bits = args.feat_bits

    def _init_gaussians(self, args):
        self.disable_prog_optim = args.disable_prog_optim
        if not self.disable_prog_optim and not self.evaluate:
            self.initial_ratio = args.initial_ratio
            self.add_times = args.add_times
            self.add_steps = args.add_steps
            min_steps = self.add_steps * self.add_times + args.post_min_steps
            if args.max_steps < min_steps:
                self.worklog.info(f"Max steps ({args.max_steps:d}) is too small for progressive optimization. Resetting to {min_steps:d}")
                args.max_steps = min_steps
        self.topk = args.topk  # Warning: Must match hardcoded value in CUDA kernel, modify with caution
        self.eps = 1e-7 if args.disable_tiles else 1e-4  # Warning: Must match hardcoded value in CUDA kernel, modify with caution
        self.init_scale = args.init_scale
        self.disable_topk_norm = args.disable_topk_norm
        self.disable_inverse_scale = args.disable_inverse_scale
        self.disable_color_init = args.disable_color_init

    def _init_loss(self, args):
        self.l1_loss_ratio = args.l1_loss_ratio
        self.l2_loss_ratio = args.l2_loss_ratio
        self.ssim_loss_ratio = args.ssim_loss_ratio

    def _init_optimization(self, args):
        self.disable_tiles = args.disable_tiles
        self.start_step = 1
        self.max_steps = args.max_steps
        self.pos_lr = args.pos_lr
        self.scale_lr = args.scale_lr
        self.rot_lr = args.rot_lr
        self.feat_lr = args.feat_lr
        self.disable_lr_schedule = args.disable_lr_schedule
        if not self.disable_lr_schedule:
            self.decay_ratio = args.decay_ratio
            self.check_decay_steps = args.check_decay_steps
            self.max_decay_times = args.max_decay_times
            self.decay_threshold = args.decay_threshold

    def _init_pos_scale_feat(self, gt_images, gsimage):
        img_h, img_w = gt_images.shape[-2:]
        xy, scale, feat = gsimage.xy, gsimage.scale, gsimage.feat
        pixel_xy = get_grid(h=img_h, w=img_w).to(dtype=xy.dtype, device=xy.device).reshape(-1, 2)
        num_pixels = img_h * img_w
        num_gaussians = xy.shape[0]
        with torch.no_grad():
            # Position
            if self.init_mode == 'gradient':
                gradients = self._compute_gmap(gt_images)
                xy.copy_(self._sample_pos(prob=gradients, pixel_xy=pixel_xy, num_pixels=num_pixels, num_gaussians=num_gaussians))
            elif self.init_mode == 'saliency':
                saliency = self._compute_smap(gt_images, path="models")
                xy.copy_(self._sample_pos(prob=saliency, pixel_xy=pixel_xy, num_pixels=num_pixels, num_gaussians=num_gaussians))
            else:
                selected = np.random.choice(num_pixels, num_gaussians, replace=False, p=None)
                xy.copy_(pixel_xy.detach().clone()[selected])
            # Scale     # TODO: allow non-uniform scale initialization based on distances between points (in xy)
            scale.fill_(self.init_scale if self.disable_inverse_scale else 1.0 / self.init_scale)
            # Feature
            feat.copy_(self._get_target_features(gt_images=gt_images, positions=xy).detach().clone())
        return gsimage

    def _sample_pos(self, prob, pixel_xy, num_pixels, num_gaussians):
        num_random = round(self.init_random_ratio * num_gaussians)
        selected_random = np.random.choice(num_pixels, num_random, replace=False, p=None)
        selected_other = np.random.choice(num_pixels, num_gaussians-num_random, replace=False, p=prob)
        return torch.cat([pixel_xy.detach().clone()[selected_random], pixel_xy.detach().clone()[selected_other]], dim=0)

    def _compute_gmap(self, gt_images):
        gy, gx = compute_image_gradients(np.power(gt_images.detach().cpu().clone().numpy(), 1.0 / self.gamma))
        g_norm = np.hypot(gy, gx).astype(np.float32)
        g_norm = g_norm / g_norm.max()
        g_norm = np.power(g_norm.reshape(-1), 2.0)
        image_gradients = g_norm / g_norm.sum()
        return image_gradients

    def _compute_smap(self, gt_images, path):
        smap = get_smap(torch.pow(gt_images.detach().clone(), 1.0 / self.gamma), path, self.smap_filter_size)
        saliency = (smap / smap.sum()).reshape(-1)
        return saliency

    def _get_target_features(self, gt_images, positions):
        with torch.no_grad():
            # gt_images [1, C, H, W]; positions [1, 1, P, 2]; top-left [-1, -1]; bottom-right [1, 1]
            target_features = F.grid_sample(gt_images.unsqueeze(0), positions[None, None, ...] * 2.0 - 1.0, align_corners=False)
            target_features = target_features[0, :, 0, :].permute(1, 0)  # [P, C]
        return target_features

    def _quantize(self, xy, rot, scale, feat):
        with torch.no_grad():
            xy.copy_(ste_quantize(xy, self.pos_bits))
            scale.copy_(ste_quantize(scale, self.scale_bits))
            rot.copy_(ste_quantize(rot, self.rot_bits))
            feat.copy_(ste_quantize(feat, self.feat_bits))
        return xy, rot, scale, feat

    def render(self, gsimage, upsample_ratio=1.):
        xy, rot, scale, feat = gsimage.xy, gsimage.rot, gsimage.scale, gsimage.feat
        img_h, img_w = gsimage.size[0].item(), gsimage.size[1].item()
        tile_bounds = ((img_w + self.block_w - 1) // self.block_w, (img_h + self.block_h - 1) // self.block_h, 1)
        with torch.no_grad():
            num_prep_runs = 2
            for _ in range(num_prep_runs):
                self.forward(gsimage, tile_bounds, upsample_ratio, benchmark=True)
            images, render_time = self.forward(gsimage, tile_bounds, upsample_ratio)
        return images, render_time

    def benchmark_render_time(self, num_reps, render_height=None):
        img_h, img_w = self.img_h, self.img_w
        if render_height is not None:
            img_h, img_w = render_height, round((float(render_height)/img_h)*img_w)
        tile_bounds = ((img_w + self.block_w - 1) // self.block_w, (img_h + self.block_h - 1) // self.block_h, 1)
        upsample_ratio = float(img_h) / self.img_h
        with torch.no_grad():
            render_time_all = np.zeros(num_reps, dtype=np.float32)
            num_prep_runs = 2
            for _ in range(num_prep_runs):
                self.forward(img_h, img_w, tile_bounds, upsample_ratio, benchmark=True)
            for rid in range(num_reps):
                render_time = self.forward(img_h, img_w, tile_bounds, upsample_ratio, benchmark=True)
                render_time_all[rid] = render_time
        return render_time_all

    # rendering function
    def forward(self, gsimage, tile_bounds=None, upsample_ratio=None, benchmark=False):
        xy, rot, scale, feat = gsimage.xy, gsimage.rot, gsimage.scale, gsimage.feat
        img_h, img_w, feat_dim = gsimage.size[0].item(), gsimage.size[1].item(), gsimage.feat_dim
        if tile_bounds is None:
            tile_bounds = ((img_w + self.block_w - 1) // self.block_w, (img_h + self.block_h - 1) // self.block_h, 1)
        scale = self._get_scale(scale=scale, upsample_ratio=upsample_ratio)
        if self.quantize:
            xy, scale, rot, feat = ste_quantize(xy, self.pos_bits), ste_quantize(
                scale, self.scale_bits), ste_quantize(rot, self.rot_bits), ste_quantize(feat, self.feat_bits)
        begin = perf_counter()
        tmp = project_gaussians_2d_scale_rot(xy, scale, rot, img_h, img_w, tile_bounds)
        xy, radii, conics, num_tiles_hit = tmp
        if not self.disable_tiles:
            enable_topk_norm = not self.disable_topk_norm
            tmp = xy, radii, conics, num_tiles_hit, feat, img_h, img_w, self.block_h, self.block_w, enable_topk_norm
            out_image = rasterize_gaussians_sum(*tmp)
        else:
            tmp = xy, conics, feat, img_h, img_w
            out_image = rasterize_gaussians_no_tiles(*tmp)
        render_time = perf_counter() - begin
        if benchmark:
            return render_time
        out_image = out_image.view(-1, img_h, img_w, feat_dim).permute(0, 3, 1, 2).contiguous()
        return out_image.squeeze(dim=0), render_time

    def _get_scale(self, scale=None, upsample_ratio=None):
        if not self.disable_inverse_scale:
            scale = 1.0 / scale
        if upsample_ratio is not None:
            scale = upsample_ratio * scale
        return scale

    def optimize(self, images, total_num_gaussians=4000, target_psnr=40.0):     # (B, C, H, W), in range (0, 1)
        device = images.device
        batsize, numchannels, img_h, img_w = images.shape
        assert images.shape[0] == 1, "Only batch size of 1 is supported"
        images = images[0]

        num_gaussians = math.ceil(self.initial_ratio * total_num_gaussians)
        max_add_num = math.ceil(float(total_num_gaussians-num_gaussians) / self.add_times)

        psnr_curr, ssim_curr, best_psnr, best_ssim = 0.0, 0.0, 0.0, 0.0
        decay_times, no_improvement_steps = 0, 0
        render_time_accum, total_time_accum = 0.0, 0.0
        lpips_final, flip_final, msssim_final = 1.0, 1.0, 0.0

        num_pixels = img_h * img_w
        tile_bounds = ((img_w + self.block_w - 1) // self.block_w, (img_h + self.block_h - 1) // self.block_h, 1)

        # initialize gaussian representations
        gsimage = GSImage(num_gaussians=num_gaussians, feat_dim=numchannels, size=(img_h, img_w), dtype=self.dtype, device=device)
        self._init_pos_scale_feat(gt_images=images, gsimage=gsimage)

        # create optimizer
        optimizer = torch.optim.Adam([{'params': gsimage.xy, 'lr': self.pos_lr},
                                    {'params': gsimage.scale, 'lr': self.scale_lr},
                                    {'params': gsimage.rot, 'lr': self.rot_lr},
                                    {'params': gsimage.feat, 'lr': self.feat_lr}])

        for step in range(self.start_step, self.max_steps+1):
            optimizer.zero_grad()
            # Rendering
            recons_images, render_time = self.forward(gsimage, tile_bounds)       # (C, H, W), more or less in (0, 1)
            render_time_accum += render_time
            # Optimization
            begin = perf_counter()
            total_loss, (l1_loss, l2_loss, ssim_loss) = self._get_total_loss(recons_images, images)
            total_loss.backward()
            optimizer.step()
            total_time_accum += (perf_counter() - begin + render_time)
            # Logging
            terminate = False
            with torch.no_grad():
                num_gaussians = gsimage.xy.shape[0]
                if step % self.eval_steps == 0:
                    if not self.disable_lr_schedule and num_gaussians == total_num_gaussians:
                        psnr_curr, ssim_curr = self._evaluate(gsimage, images)
                        terminate, (best_psnr, best_ssim, no_improvement_steps, decay_times) \
                            = self._lr_schedule(psnr_curr, ssim_curr, best_psnr, best_ssim, no_improvement_steps, decay_times, optimizer)
                        if psnr_curr >= target_psnr:
                            terminate = True
                if not self.disable_prog_optim and step % self.add_steps == 0 and num_gaussians < total_num_gaussians:
                    add_num = min(max_add_num, total_num_gaussians-num_gaussians)
                    gsimage = self._add_gaussians(images[0], gsimage, tile_bounds, add_num)

                    # Update optimizer
                    optimizer = torch.optim.Adam([{'params': gsimage.xy, 'lr': self.pos_lr},
                                                    {'params': gsimage.scale, 'lr': self.scale_lr},
                                                    {'params': gsimage.rot, 'lr': self.rot_lr},
                                                    {'params': gsimage.feat, 'lr': self.feat_lr}])
                if terminate and step >= self.min_steps:
                    break
        recons_images = self.forward(gsimage)[0]
        psnr_curr, ssim_curr = self._evaluate(gsimage, images)
        return gsimage, recons_images, {"psnr": psnr_curr, "ssim": ssim_curr}

    def _get_total_loss(self, images, gt_images):
        total_loss = 0
        if self.l1_loss_ratio > 1e-7:
            l1_loss = self.l1_loss_ratio * F.l1_loss(images, gt_images)
            total_loss += l1_loss
        else:
            l1_loss = None
        if self.l2_loss_ratio > 1e-7:
            l2_loss = self.l2_loss_ratio * F.mse_loss(images, gt_images)
            total_loss += l2_loss
        else:
            l2_loss = None
        if self.ssim_loss_ratio > 1e-7:
            ssim_loss = self.ssim_loss_ratio * (1 - fused_ssim(images.unsqueeze(0), gt_images.unsqueeze(0)))
            total_loss += ssim_loss
        else:
            ssim_loss = None
        return total_loss, (l1_loss, l2_loss, ssim_loss)

    def _evaluate(self, gsimage, gt_images):
        tile_bounds = ((gsimage.img_w + self.block_w - 1) // self.block_w, (gsimage.img_h + self.block_h - 1) // self.block_h, 1)
        images = torch.pow(torch.clamp(self.forward(gsimage, tile_bounds=tile_bounds)[0], 0.0, 1.0), 1.0 / self.gamma)
        gt_images = torch.pow(gt_images, 1.0 / self.gamma)
        psnr = get_psnr(images, gt_images).item()
        ssim = fused_ssim(images.unsqueeze(0), gt_images.unsqueeze(0)).item()
        return psnr, ssim

    def _evaluate_extra(self):
        images = torch.pow(torch.clamp(self._render_images(upsample=False), 0.0, 1.0), 1.0/self.gamma)[None, ...]
        gt_images = torch.pow(self.gt_images, 1.0/self.gamma)[None, ...]
        msssim_metric = MS_SSIM(data_range=1.0, size_average=True, channel=self.feat_dim).to(device=self.device).eval()
        self.msssim_final = msssim_metric(images, gt_images).item()
        lpips_metric = LPIPS(net='alex').to(device=self.device).eval()
        flip_metric = LDRFLIPLoss().to(device=self.device).eval()
        num_channels = 1 if self.feat_dim < 3 else 3
        self.lpips_final = lpips_metric(images[:, :num_channels], gt_images[:, :num_channels]).item()
        if self.feat_dim >= 3:
            self.flip_final = flip_metric(images[:, :3], gt_images[:, :3]).item()

    def _lr_schedule(self, psnr_curr, ssim_curr, best_psnr, best_ssim, no_improvement_steps=0, decay_times=0, optimizer=None):
        if (psnr_curr <= best_psnr + 100 * self.decay_threshold or ssim_curr <= best_ssim + self.decay_threshold):
            no_improvement_steps += self.eval_steps
            if no_improvement_steps >= self.check_decay_steps:
                no_improvement_steps = 0
                decay_times += 1
                if decay_times > self.max_decay_times:
                    return True, (best_psnr, best_ssim, no_improvement_steps, decay_times)
                for param_group in optimizer.param_groups:
                    param_group['lr'] /= self.decay_ratio
                print(f"Learning rate decayed by {self.decay_ratio:.1f}")
        else:
            best_psnr = psnr_curr
            best_ssim = ssim_curr
            no_improvement_steps = 0
            decay_times = 0
        return False, (best_psnr, best_ssim, no_improvement_steps, decay_times)

    def _add_gaussians(self, gt_images, gsimage, tile_bounds, add_num):
        xy, rot, scale, feat = gsimage.xy, gsimage.rot, gsimage.scale, gsimage.feat
        img_h, img_w = gt_images.shape[-2:]
        device = gt_images.device
        dtype = gt_images.dtype
        pixel_xy = get_grid(h=img_h, w=img_w).to(dtype=dtype, device=device).reshape(-1, 2)
        num_pixels = img_h * img_w
        if add_num <= 0:
            return
        raw_images, _ = self.forward(gsimage, tile_bounds)
        images = torch.pow(torch.clamp(raw_images, 0.0, 1.0), 1.0 / self.gamma)
        gt_images = torch.pow(gt_images, 1.0 / self.gamma)

        kernel_size = round(np.sqrt(img_h * img_w) // 400)
        if kernel_size >= 1:
            kernel_size = max(3, kernel_size)
            kernel_size = kernel_size + 1 if kernel_size % 2 == 0 else kernel_size
            gt_images = gaussian_blur(img=gt_images, kernel_size=kernel_size)

        diff_map = (gt_images - images).detach().clone()
        error_map = torch.pow(torch.abs(diff_map).mean(dim=0).reshape(-1), 2.0)
        sample_prob = (error_map / error_map.sum()).cpu().numpy()
        selected = np.random.choice(num_pixels, add_num, replace=False, p=sample_prob)

        # New Gaussians
        new_xy = pixel_xy.detach().clone()[selected]
        new_scale = torch.ones(add_num, 2, dtype=self.dtype, device=device)
        init_scale = self.init_scale
        new_scale.fill_(init_scale if self.disable_inverse_scale else 1.0 / init_scale)
        new_rot = torch.zeros(add_num, 1, dtype=self.dtype, device=device)
        new_feat = diff_map.permute(1, 2, 0).reshape(-1, gsimage.feat_dim)[selected]

        # Old Gaussians
        old_xy = xy.detach().clone()
        old_scale = scale.detach().clone()
        old_rot = rot.detach().clone()
        old_feat = feat.detach().clone()

        # Update trainable parameters
        all_xy = torch.cat([old_xy, new_xy], dim=0)
        all_scale = torch.cat([old_scale, new_scale], dim=0)
        all_rot = torch.cat([old_rot, new_rot], dim=0)
        all_feat = torch.cat([old_feat, new_feat], dim=0)

        xy = nn.Parameter(all_xy, requires_grad=True)
        scale = nn.Parameter(all_scale, requires_grad=True)
        rot = nn.Parameter(all_rot, requires_grad=True)
        feat = nn.Parameter(all_feat, requires_grad=True)

        gsimage.xy, gsimage.scale, gsimage.rot, gsimage.feat = xy, scale, rot, feat

        return gsimage
