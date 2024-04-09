import os

os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'

import numpy as np
import torch
import torch.nn.functional as F
import cv2 as cv
import trimesh
import yaml
import tqdm

import smplx
from network.volume import CanoBlendWeightVolume
from utils.renderer import Renderer
import config

from easymocap.bodymodel.lbs import batch_rodrigues


def save_pos_map(pos_map, path):
    mask = np.linalg.norm(pos_map, axis = -1) > 0.
    positions = pos_map[mask]
    print('Point nums %d' % positions.shape[0])
    pc = trimesh.PointCloud(positions)
    pc.export(path)


def interpolate_lbs(pts, vertices, faces, vertex_lbs):
    from utils.posevocab_custom_ops.nearest_face import nearest_face_pytorch3d
    from utils.geo_util import barycentric_interpolate
    dists, indices, bc_coords = nearest_face_pytorch3d(
        torch.from_numpy(pts).to(torch.float32).cuda()[None],
        torch.from_numpy(vertices).to(torch.float32).cuda()[None],
        torch.from_numpy(faces).to(torch.int64).cuda()
    )
    # print(dists.mean())
    lbs = barycentric_interpolate(
        vert_attris = vertex_lbs[None].to(torch.float32).cuda(),
        faces = torch.from_numpy(faces).to(torch.int64).cuda()[None],
        face_ids = indices,
        bc_coords = bc_coords
    )
    return lbs[0].cpu().numpy()


map_size = 1024


if __name__ == '__main__':
    from argparse import ArgumentParser
    import importlib

    arg_parser = ArgumentParser()
    arg_parser.add_argument('-c', '--config_path', type = str, help = 'Configuration file path.')
    args = arg_parser.parse_args()

    opt = yaml.load(open(args.config_path, encoding = 'UTF-8'), Loader = yaml.FullLoader)
    dataset_module = opt['train'].get('dataset', 'MvRgbDatasetAvatarReX')
    MvRgbDataset = importlib.import_module('dataset.dataset_mv_rgb').__getattribute__(dataset_module)
    dataset = MvRgbDataset(**opt['train']['data'])
    data_dir, frame_list = dataset.data_dir, dataset.pose_list

    # os.makedirs(data_dir + '/smpl_pos_map', exist_ok = True)
    os.makedirs( 'smpl_pos_map', exist_ok = True)
    smplx_mask=opt.get("smplx_mask",False)


    cano_renderer = Renderer(map_size, map_size, shader_name = 'vertex_attribute')

    smpl_model = smpl_model = smplx.SMPLXLayer(model_path='./smpl_files/smplx',
                                           gender='neutral',
                                           use_compressed=False,
                                           use_face_contour=True,
                                           num_expression_coeffs=100)
    # smpl_data = np.load(data_dir + '/smpl_params.npz')
    # smpl_data = {k: torch.from_numpy(v.astype(np.float32)) for k, v in smpl_data.items()}
    smpl_data = {}
    imgs_keys=list(dataset.smpl_data.keys())
    first_img_index=imgs_keys[0]
    for key in dataset.smpl_data[first_img_index]:
        smpl_data[key]=torch.zeros(len(imgs_keys), *dataset.smpl_data[first_img_index][key].shape, dtype=torch.float32)
    index =0
    for i in dataset.smpl_data:
        for j in dataset.smpl_data[i]:
            smpl_data[j][index]=dataset.smpl_data[i][j]
        index=index+1

    betas = np.load(data_dir + '/cano_betas.npy')
    cano_smpl_betas = torch.from_numpy(betas).float()
    with torch.no_grad():
        cano_smpl = smpl_model.forward(
            betas = cano_smpl_betas,
            global_orient = batch_rodrigues(config.cano_smpl_global_orient.reshape(-1, 3))[None],
            transl = config.cano_smpl_transl[None],
            body_pose = batch_rodrigues(config.cano_smpl_body_pose.reshape(-1, 3))[None]
        )
        cano_smpl_v = cano_smpl.vertices[0].cpu().numpy()
        cano_center = 0.5 * (cano_smpl_v.min(0) + cano_smpl_v.max(0))
        cano_smpl_v_min = cano_smpl_v.min()
        smpl_faces = smpl_model.faces.astype(np.int64)

    if os.path.exists(data_dir + '/template.ply'):
        print('# Loading template from %s' % (data_dir + '/template.ply'))
        template = trimesh.load(data_dir + '/template.ply', process = False)
        using_template = True
    else:
        print(f'# Cannot find template.ply from {data_dir}, using SMPL-X as template')
        template = trimesh.Trimesh(cano_smpl_v, smpl_faces, process = False)
        using_template = False

    cano_smpl_v = template.vertices.astype(np.float32)
    smpl_faces = template.faces.astype(np.int64)
    if smplx_mask:
        smplx_flame=np.load("./smpl_files/SMPL-X__FLAME_vertex_ids.npy")
        smpl_v_mask=np.zeros(cano_smpl_v.shape[0])+1
        for i in range(smplx_flame.shape[0]):
            smpl_v_mask[smplx_flame[i]]=0
        smpl_faces_mask=[]
        for i in range(smpl_faces.shape[0]):
            if smpl_v_mask[smpl_faces[i][0]] and smpl_v_mask[smpl_faces[i][1]] and smpl_v_mask[smpl_faces[i][2]]:
                smpl_faces_mask=smpl_faces_mask+[smpl_faces[i]]
        cano_smpl_v_dup_full = cano_smpl_v[smpl_faces.reshape(-1)]
        cano_smpl_n_dup_full = template.vertex_normals.astype(np.float32)[smpl_faces.reshape(-1)]
        smpl_faces_full=smpl_faces
        smpl_faces=np.array(smpl_faces_mask)
    cano_smpl_v_dup = cano_smpl_v[smpl_faces.reshape(-1)]
    cano_smpl_n_dup = template.vertex_normals.astype(np.float32)[smpl_faces.reshape(-1)]

    # define front & back view matrices
    front_mv = np.identity(4, np.float32)
    front_mv[:3, 3] = -cano_center + np.array([0, 0, -10], np.float32)
    front_mv[1:3] *= -1

    back_mv = np.identity(4, np.float32)
    rot_y = cv.Rodrigues(np.array([0, np.pi, 0], np.float32))[0]
    back_mv[:3, :3] = rot_y
    back_mv[:3, 3] = -rot_y @ cano_center + np.array([0, 0, -10], np.float32)
    back_mv[1:3] *= -1

    # render canonical smpl position maps
    cano_renderer.set_model(cano_smpl_v_dup, cano_smpl_v_dup)
    cano_renderer.set_camera(front_mv)
    front_cano_pos_map = cano_renderer.render()[:, :, :3]

    if smplx_mask:
        cano_renderer.set_model(cano_smpl_v_dup_full, cano_smpl_v_dup_full)
    cano_renderer.set_camera(back_mv)
    back_cano_pos_map = cano_renderer.render()[:, :, :3]
    back_cano_pos_map = cv.flip(back_cano_pos_map, 1)
    cano_pos_map = np.concatenate([front_cano_pos_map, back_cano_pos_map], 1)
    # cv.imwrite(data_dir + '/smpl_pos_map/cano_smpl_pos_map.exr', cano_pos_map)
    cv.imwrite('smpl_pos_map/cano_smpl_pos_map.exr', cano_pos_map)

    # render canonical smpl normal maps
    cano_renderer.set_model(cano_smpl_v_dup, cano_smpl_n_dup)
    cano_renderer.set_camera(front_mv)
    front_cano_nml_map = cano_renderer.render()[:, :, :3]

    if smplx_mask:
        cano_renderer.set_model(cano_smpl_v_dup_full, cano_smpl_n_dup_full)
    cano_renderer.set_camera(back_mv)
    back_cano_nml_map = cano_renderer.render()[:, :, :3]
    back_cano_nml_map = cv.flip(back_cano_nml_map, 1)
    cano_nml_map = np.concatenate([front_cano_nml_map, back_cano_nml_map], 1)
    # cv.imwrite(data_dir + '/smpl_pos_map/cano_smpl_nml_map.exr', cano_nml_map)
    cv.imwrite('smpl_pos_map/cano_smpl_nml_map.exr', cano_nml_map)

    body_mask = np.linalg.norm(cano_pos_map, axis = -1) > 0.
    # cv.imwrite(data_dir + 'smpl_pos_map/%08d.exr' % pose_idx, live_pos_map)
    cv.imwrite('smpl_pos_map/body_mask.png', body_mask * 255)

    cano_pts = cano_pos_map[body_mask]
    if smplx_mask:
        smpl_faces=smpl_faces_full
    if using_template:
        weight_volume = CanoBlendWeightVolume(data_dir + '/cano_weight_volume.npz')
        pts_lbs = weight_volume.forward_weight(torch.from_numpy(cano_pts)[None].cuda())[0]
    else:
        pts_lbs = interpolate_lbs(cano_pts, cano_smpl_v, smpl_faces, smpl_model.lbs_weights)
        pts_lbs = torch.from_numpy(pts_lbs).cuda()
    # np.save(data_dir + '/smpl_pos_map/init_pts_lbs.npy', pts_lbs.cpu().numpy())
    np.save('smpl_pos_map/init_pts_lbs.npy', pts_lbs.cpu().numpy())

    inv_cano_smpl_A = torch.linalg.inv(cano_smpl.A).cuda()
    body_mask = torch.from_numpy(body_mask).cuda()
    cano_pts = torch.from_numpy(cano_pts).cuda()
    pts_lbs = pts_lbs.cuda()

    for pose_idx in tqdm.tqdm(range(len(frame_list)), desc = 'Generating positional maps...'):
        with torch.no_grad():
            live_smpl_woRoot = smpl_model.forward(
                betas = smpl_data['betas'][pose_idx],
                global_orient = smpl_data['global_orient'][pose_idx],
                transl = smpl_data['transl'][pose_idx],
                body_pose = smpl_data['body_pose'][pose_idx],
                jaw_pose = smpl_data['jaw_pose'][pose_idx],
                expression = smpl_data['expression'][pose_idx],
                left_hand_pose = smpl_data['left_hand_pose'][pose_idx],
                right_hand_pose = smpl_data['right_hand_pose'][pose_idx],
                leye_pose = smpl_data['leye_pose'][pose_idx],
                reye_pose = smpl_data['reye_pose'][pose_idx]
            )

        cano2live_jnt_mats_woRoot = torch.matmul(live_smpl_woRoot.A.cuda(), inv_cano_smpl_A)[0]
        pt_mats = torch.einsum('nj,jxy->nxy', pts_lbs, cano2live_jnt_mats_woRoot)
        live_pts = torch.einsum('nxy,ny->nx', pt_mats[..., :3, :3], cano_pts) + pt_mats[..., :3, 3]
        live_pos_map = torch.zeros((map_size, 2 * map_size, 3)).to(live_pts)
        live_pos_map[body_mask] = live_pts
        live_pos_map = F.interpolate(live_pos_map.permute(2, 0, 1)[None], None, [0.5, 0.5], mode = 'nearest')[0]
        live_pos_map = live_pos_map.permute(1, 2, 0).cpu().numpy()

        # cv.imwrite(data_dir + 'smpl_pos_map/%08d.exr' % frame[pose_idx], live_pos_map)
        cv.imwrite('smpl_pos_map/%08d.exr' % frame_list[pose_idx], live_pos_map)


