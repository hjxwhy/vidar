# TRI-VIDAR - Copyright 2022 Toyota Research Institute.  All rights reserved.

import random
import os
import numpy as np
import torch

from vidar.arch.losses.MultiCamPhotometricLoss import MultiCamPhotometricLoss
from vidar.arch.models.BaseModel import BaseModel
from vidar.arch.networks.layers.fsm.camera import Camera
from vidar.arch.networks.layers.fsm.pose import Pose
from vidar.arch.networks.layers.fsm.utils import \
    CameraNormalizer, flip_batch_input, flip_output, filter_dict, coords_from_motion, mask_from_coords, mask_from_mask
from vidar.utils.depth import depth2inv, inv2depth
from vidar.utils.types import is_list, is_seq
from vidar.utils.read import read_image

def split_batch(tensor, n=1):
    """Split a tensor batch-wise"""
    if is_list(tensor):
        split = [split_batch(t, n=n) for t in tensor]
        return list(map(list, zip(*split)))
    return torch.split(tensor, split_size_or_sections=n, dim=0)


def global_cameras(intrinsics, pose, pose_context, hw=None):
    """Create global cameras for target and source poses + target intrinsics"""
    cam = camera_from_intrinsics_pose(pose, intrinsics, hw=hw)
    cam_context = camera_from_intrinsics_pose(pose_context, intrinsics, pose, hw=hw)
    return cam, cam_context


def camera_from_intrinsics_pose(pose, intrinsics, orig_pose=None, hw=None):
    """
    Create one or more cameras from pose and intrinsics

    Parameters
    ----------
    pose : torch.Tensor or list[torch.Tensor]
        Poses to be used [B,4,4]
    intrinsics : torch.Tensor or list[torch.Tensor]
        Intrinsics to be used [B,3,3]
    orig_pose : torch.Tensor or list[torch.Tensor]
        Original poses from which pose is generated [B,4,4]
    hw : tuple
        Camera image dimensions

    Returns
    -------
    camera : Camera
        Camera instance created from the input
    """
    # If pose is a sequence, do it for each one
    if is_seq(pose):
        # If intrinsics is not a sequence, repeat it
        if not is_seq(intrinsics):
            intrinsics = [intrinsics] * len(pose)
        # If orig pose is not a sequence, repeat it
        if not is_seq(orig_pose):
            orig_pose = [orig_pose] * len(pose)
        # Recursive loop for each item
        return [camera_from_intrinsics_pose(p, i, o, hw=hw)
                for p, i, o in zip(pose, intrinsics, orig_pose)]
    # Compound original pose if available
    if orig_pose is not None:
        pose = Pose(orig_pose) @ Pose(pose).inverse()
    # Return camera
    return Camera(K=intrinsics, Twc=pose, hw=hw)


class FSMModel(BaseModel):
    """
    Full Surround Monodepth (FSM) model (https://arxiv.org/abs/2104.00152)

    Parameters
    ----------
    cfg : Config
        Configuration with parameters
    """
    def __init__(self, cfg, **kwargs):
        super().__init__(cfg)

        self.rotation_mode = 'euler'
        self.flip_lr_prob = 0.0
        self.upsample_depth_maps = False

        norm_focal = []
        self.focal = None if len(norm_focal) == 0 else CameraNormalizer(focal=norm_focal)
        self.cameras = [[1],[5],[6],[7],[8],[9]]
        mask = torch.stack([torch.from_numpy(np.array(read_image(os.path.join(
                        './data/masks/ddad', '%02d.png' % i[0])))) for i in self.cameras])
        self.mask = mask.unsqueeze(1)
        pairs = [
            [0, 1], [1, 0],
            [0, 2], [2, 0],
            [1, 4], [4, 1],
            [2, 3], [3, 2],
            [3, 5], [5, 3],
            [4, 5], [5, 4],
        ]
        self.neighbors = torch.tensor([[1, 3, 0, 5, 2, 4], [2, 0, 4, 1, 5, 3]])

        stereo_weight = 0.1
        stereo_context = True
        gt_pose = False

        self.multicam_loss = MultiCamPhotometricLoss(**kwargs)
        self.pairs = pairs
        self.stereo_weight = stereo_weight
        self.gt_pose = gt_pose
        self.stereo_context = stereo_context

        self._input_keys = ['rgb', 'rgb_context', 'intrinsics', 'extrinsics',
                            'pose_context', 'filename']
        self.networks = torch.nn.ModuleDict()

    def compute_depth_net(self, batch, force_flip=False):
        """Computes inverse depth maps from single images"""
        # Randomly flip and estimate inverse depth maps
        flag_flip_lr = random.random() < self.flip_lr_prob if self.training else force_flip
        output = self.depth_net_flipping(batch, flag_flip_lr)
        if self.focal is not None:
            output['inv_depths'] = self.focal.unormalize(output['inv_depths'])
        # Return inverse depth maps
        return output

    def compute_pose_net(self, image, contexts):
        """Compute poses from image and a sequence of context images"""
        pose_vec = self.networks['pose'](image, contexts)
        return [Pose.from_vec(pose_vec[:, i], self.rotation_mode)
                for i in range(pose_vec.shape[1])]

    def depth_net_flipping(self, batch, flip):

        # Which keys are being passed to the depth network
        batch_input = {key: batch[key] for key in filter_dict(batch, self._input_keys)}
        if self.focal is not None:
            batch_input['rgb'] = self.focal.normalize(batch_input['rgb'], batch['intrinsics'])
        if flip:
            # Run depth network with flipped inputs
            output = self.networks['depth'](**flip_batch_input(batch_input))
            # Flip output back if training
            if self.training:
                output = flip_output(output)
        else:
            # Run depth network
            output = self.networks['depth'](**batch_input)
        return output

    def forward2(self, batch, return_logs=False, force_flip=False):
        # Generate inverse depth predictions
        depth_output = self.compute_depth_net(batch, force_flip=force_flip)
        # Generate pose predictions if available
        pose_output = None
        if 'rgb_context' in batch and self.networks['pose'] is not None:
            pose_output = self.compute_pose_net(
                batch['rgb'], batch['rgb_context'])
        # Return output dictionary
        return {
            **depth_output,
            'poses': pose_output,
        }

    def forward(self, batch, return_logs=False, progress=0.0, **kwargs):

        new_batch = {}
        new_batch['rgb'] = batch['rgb'][0]  # [B, N, C, H, W]
        new_batch['mask'] = batch['mask']
        if self.training:
            new_batch['rgb_context'] = [batch['rgb'][1], batch['rgb'][-1]]
            new_batch['pose'] = batch['pose'][0]
            new_batch['pose_context'] = [batch['pose'][1], batch['pose'][-1]]
        new_batch['intrinsics'] = batch['intrinsics'][0]
        new_batch['filename'] = batch['filename']
        batch = new_batch

        B, N, C, H, W = batch['rgb'].shape
        batch['rgb'] = batch['rgb'].reshape([-1, *batch['rgb'].shape[2:]])
        batch['intrinsics'] = batch['intrinsics'].reshape([-1, *batch['intrinsics'].shape[2:]])
        batch['mask'] = batch['mask'].reshape([-1, *batch['mask'].shape[2:]])

        if self.training:
            # batch['rgb'] = batch['rgb'][0]
            batch['rgb_context'] = [b.reshape([-1, *b.shape[2:]]) for b in batch['rgb_context']]
            batch['pose'] = batch['pose'].reshape([-1, *batch['pose'].shape[2:]])
            batch['pose_context'] = [b.reshape([-1, *b.shape[2:]]) for b in batch['pose_context']]
            # batch['intrinsics'] = batch['intrinsics'].reshape([-1, *batch['intrinsics'].shape[2:]])

        # depth(four scale depth predict, 0 is the final): [tensor[6, 1, H, W]*4]
        if self.networks['depth'] is not None:
            output_self_sup = self.forward2(batch)
            depth = inv2depth(output_self_sup['inv_depths'])
        else:
            output_self_sup = {}
            depth = batch['depth']

        if not self.training:
            output_new = {
                'predictions': {
                    'depth': {
                        0: [1. / d for d in output_self_sup['inv_depths']]
                    }
                }
            }
            output_self_sup = output_new
            return output_self_sup

        rgb = batch['rgb']
        rgb_context = batch['rgb_context']
        intrinsics = batch['intrinsics']
        pose = batch['extrinsics'] if 'extrinsics' in batch else batch['pose']

        pose_context_gt = batch['pose_context']
        if self.gt_pose:
            pose_context = batch['pose_context']
        else:
            pose_context = output_self_sup['poses']

        for i in range(len(pose_context)):
            pose_context[i] = pose_context[i].mat  # [6, 4, 4]

        # # rgb_i: tuple(tensor[1, 3, H, W]*6), rgb_context_i:[[tensor[1, 3, H, W] * 2]*6]
        # rgb_i, rgb_context_i = split_batch(rgb), split_batch(rgb_context)
        # # pose_i: tuple(tensor[1, 4, 4] * 6), pose_context_i:[[tensor[1, 4, 4] * 2]*6]
        # pose_i, pose_context_i = split_batch(pose), split_batch(pose_context)
        # # intrinsics_i: tuple(tensor[1, 3, 3] * 6), inv_depth_i(4 is different scale): [[tensor[1, 1, H, W]*4]*6]
        # intrinsics_i, inv_depth_i = split_batch(intrinsics), depth2inv(split_batch(depth))
        # # cam_i: [Camera * 6], cam_context_i: [[Camera * 2] * 6]
        # cam_i, cam_context_i = global_cameras(intrinsics_i, pose_i, pose_context_i, hw=rgb.shape[2:])
        #
        # _, pose_context_i_gt = split_batch(pose), split_batch(pose_context_gt)
        # _, cam_context_i_gt = global_cameras(intrinsics_i, pose_i, pose_context_i_gt, hw=rgb.shape[2:])
        #
        # n_tgt = len(rgb_i)
        #
        # # mono_coords: [[[tensor[1, 2, H, W] * 4] * 2] * 6]
        # mono_coords = [coords_from_motion(
        #     cam_context_i[tgt], inv2depth(inv_depth_i[tgt]), cam_i[tgt])
        #     for tgt in range(n_tgt)]
        # # mono_masks: [[[tensor[1, 2, H, W] * 4] * 2] * 6]
        # mono_masks = [mask_from_coords(coords) for coords in mono_coords]
        #
        # filename = batch['filename']
        # try:
        #     filename = ['camera' + f[0].split('/')[-2][-3:]+ '_mask.npy' for f in filename]
        #     # filename = ['camera' + f.split('/')[-2][-3:]+ '_mask.npy' for f in filename]
        #     masks = [torch.tensor(np.load(f)).unsqueeze(0).unsqueeze(0) for f in filename]
        #     for tgt in range(n_tgt):
        #         for i in range(len(mono_masks[tgt])):
        #             for j in range(len(mono_masks[tgt][i])):
        #                 for k in range(len(mono_masks[tgt][i][j])):
        #                     # write_image('debug/camera_%d/mask_%d_%d_%d.png' % (tgt, i, j, k),
        #                     #             mono_masks[tgt][i][j][k])
        #                     resized_mask = torch.nn.functional.interpolate(
        #                         masks[tgt], mono_masks[tgt][i][j][k].shape[1:], mode='nearest').squeeze(0).bool()
        #                     mono_masks[tgt][i][j][k] *= resized_mask.to(mono_masks[tgt][i][j][k].device)
        #     with_masks = True
        # except:
        #     with_masks = False
        #     pass
        # cam:

        mono = []
        outputs = []
        # pose: [B*N, 4, 4], intrinsics: [B*N, 3, 3], pose_context:[B*N, 4, 4] * 2
        mask = batch['mask']
        cam, cam_context = global_cameras(intrinsics, pose, pose_context)
        # mono_coords:
        mono_coords = coords_from_motion(cam_context, depth, cam)
        # mono_masks: [[tensor[6, 2, H, W] * 4] * 2]
        mono_masks = [mask_from_mask(mask, coords) for coords in mono_coords]
        output = self.multicam_loss(rgb, rgb_context, inv2depth(depth), cam, cam_context, with_mask=mono_masks)
        mono.append(output['loss'])

        ################
        # ind = (torch.arange(0, B) * N).unsqueeze(-1).unsqueeze(-1) + self.neighbors.unsqueeze(0)
        # left_ind = ind[:, 0, :].flatten().long()
        # right_ind = ind[:, 1, :].flatten().long()

        # left_cam = Camera(K=intrinsics[left_ind], Twc=pose[left_ind])
        # right_cam = Camera(K=intrinsics[right_ind], Twc=pose[right_ind])
        # cam_context = [left_cam, right_cam]
        # mono_coords = coords_from_motion(cam_context, depth, cam)
        # mask_ = [mask[left_ind], mask[right_ind]]
        # mono_masks = [mask_from_mask(m, coords) for m, coords in zip(mask_, mono_coords)]
        # rgb_context = [rgb[left_ind], rgb[right_ind]]
        # output = self.multicam_loss(rgb, rgb_context, inv2depth(depth), cam, cam_context, with_mask=mono_masks)
        # mono.append(self.stereo_weight * output['loss'])
        # output = self.multicam_loss(depth[0], [depth[0][left_ind].detach(), depth[0][right_ind].detach()], inv2depth(depth), cam, cam_context, with_mask=mono_masks)
        # mono.append(output['loss'])

        # print(cam_context[0].Twc.shape, cam_context[0].Twc[left_ind].shape)
        # left_front = Camera(K=intrinsics[left_ind], Twc=Pose(cam_context[0].Twc[left_ind]))
        # left_back = Camera(K=intrinsics[left_ind], Twc=Pose(cam_context[1].Twc[left_ind]))
        # cam_context = [left_front, left_back]
        # mono_coords = coords_from_motion(cam_context, depth, cam)
        # # mask_ = [mask[left_ind], mask[right_ind]]
        # mono_masks = [mask_from_mask(m, coords) for m, coords in zip(mask_, mono_coords)]
        # rgb_context = [batch['rgb_context'][0][left_ind], batch['rgb_context'][0][left_ind]]
        # output = self.multicam_loss(rgb, rgb_context, inv2depth(depth), cam, cam_context, with_mask=mono_masks)
        # mono.append(output['loss'])


        # right_front = Camera(K=intrinsics[right_ind], Twc=Pose(cam_context[0].Twc[right_ind]))
        # right_back = Camera(K=intrinsics[right_ind], Twc=Pose(cam_context[1].Twc[right_ind]))
        # cam_context = [right_front, right_back]
        # mono_coords = coords_from_motion(cam_context, depth, cam)
        # # mask_ = [mask[left_ind], mask[right_ind]]
        # mono_masks = [mask_from_mask(m, coords) for m, coords in zip(mask_, mono_coords)]
        # rgb_context = [batch['rgb_context'][0][right_ind], batch['rgb_context'][0][right_ind]]
        # output = self.multicam_loss(rgb, rgb_context, inv2depth(depth), cam, cam_context, with_mask=mono_masks)
        # mono.append(output['loss'])
        ################

        # mono = []
        # outputs = []
        # # rgb_i: tuple(tensor[1, 3, H, W]*6), rgb_context_i:[[tensor[1, 3, H, W] * 2]*6]
        # # inv_depth_i: [[tensor[1, 1, H, W]*4]*6]
        # # cam_i: [Camera * 6], cam_context_i: [[Camera * 2] * 6]
        # # mono_masks: [[[tensor[1, 2, H, W] * 4] * 2] * 6]
        # for tgt in range(n_tgt):
        #     output = self.multicam_loss(
        #         rgb_i[tgt], rgb_context_i[tgt], inv_depth_i[tgt],
        #         cam_i[tgt], cam_context_i[tgt], with_mask=mono_masks[tgt])
        #     if not torch.isnan(output['loss']):
        #         mono.append(output['loss'])
        #         outputs.append(output)

        stereo = []
        # if not self.stereo_context and self.stereo_weight > 0:
        #
        #     stereo_coords = [coords_from_motion(
        #         [cam_i[src]], inv2depth(inv_depth_i[tgt]), cam_i[tgt])
        #         for tgt, src in self.pairs]
        #     stereo_masks = [mask_from_coords(coords) for coords in stereo_coords]
        #
        #     if with_masks:
        #         for tgt in range(len(self.pairs)):
        #             for i in range(len(stereo_masks[tgt])):
        #                 for j in range(len(stereo_masks[tgt][i])):
        #                     for k in range(len(stereo_masks[tgt][i][j])):
        #                         hw = stereo_masks[tgt][i][j][k].shape[1:]
        #                         h_st, h_fn = int(0.15 * hw[0]), int(0.75 * hw[0])
        #                         stereo_masks[tgt][i][j][k][:, :h_st] = 0
        #                         stereo_masks[tgt][i][j][k][:, h_fn:] = 0
        #
        #     for k, (tgt, src) in enumerate(self.pairs):
        #         output = self.multicam_loss(
        #             rgb_i[tgt], [rgb_i[src]], inv_depth_i[tgt],
        #             cam_i[tgt], [cam_i[src]], with_mask=stereo_masks[k], automask=False)
        #         if not torch.isnan(output['loss']):
        #             stereo.append(self.stereo_weight * output['loss'])
        #
        # elif self.stereo_context and self.stereo_weight > 0:
        #
        #     stereo_coords = [coords_from_motion(
        #         [cam_i[src]] + cam_context_i[src], inv2depth(inv_depth_i[tgt]), cam_i[tgt])
        #         for tgt, src in self.pairs]
        #     stereo_masks = [mask_from_coords(coords) for coords in stereo_coords]
        #
        #     for tgt in range(len(self.pairs)):
        #         for i in range(len(stereo_masks[tgt])):
        #             for j in range(len(stereo_masks[tgt][i])):
        #                 for k in range(len(stereo_masks[tgt][i][j])):
        #                     hw = stereo_masks[tgt][i][j][k].shape[1:]
        #                     h_st, h_fn = int(0.15 * hw[0]), int(0.75 * hw[0])
        #                     stereo_masks[tgt][i][j][k][:, :h_st] = 0
        #                     stereo_masks[tgt][i][j][k][:, h_fn:] = 0
        #
        #     for k, (tgt, src) in enumerate(self.pairs):
        #         output = self.multicam_loss(
        #             rgb_i[tgt], [rgb_i[src]] + rgb_context_i[src], inv_depth_i[tgt],
        #             cam_i[tgt], [cam_i[src]] + cam_context_i[src], with_mask=stereo_masks[k], automask=False)
        #         if not torch.isnan(output['loss']):
        #             stereo.append(self.stereo_weight * output['loss'])
        # mono.append(output['loss'])
        losses = mono + stereo
        loss = sum(losses) / len(losses)
        output_self_sup['loss'] = loss.unsqueeze(0)

        return {
            **output_self_sup,
            'metrics': {},
        }
