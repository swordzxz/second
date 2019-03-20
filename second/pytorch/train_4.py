import os
import pathlib
import pickle
import shutil
import time
from functools import partial
import json
import fire
import numpy as np
import torch
import random
from google.protobuf import text_format
from tensorboardX import SummaryWriter

import torchplus
import second.data.kitti_common as kitti
from second.builder import target_assigner_builder, voxel_builder
from second.data.preprocess import merge_second_batch
from second.protos import pipeline_pb2
from second.pytorch.builder import (box_coder_builder, input_reader_builder,
                                      lr_scheduler_builder, optimizer_builder,
                                      second_builder)
from second.utils.eval import get_coco_eval_result, get_official_eval_result
from second.utils.progress_bar import ProgressBar
from second.data.preprocess import prep_pointcloud

def _get_pos_neg_loss(cls_loss, labels):
    # cls_loss: [N, num_anchors, num_class]
    # labels: [N, num_anchors]
    batch_size = cls_loss.shape[0]
    if cls_loss.shape[-1] == 1 or len(cls_loss.shape) == 2:
        cls_pos_loss = (labels > 0).type_as(cls_loss) * cls_loss.view(
            batch_size, -1)
        cls_neg_loss = (labels == 0).type_as(cls_loss) * cls_loss.view(
            batch_size, -1)
        cls_pos_loss = cls_pos_loss.sum() / batch_size
        cls_neg_loss = cls_neg_loss.sum() / batch_size
    else:
        cls_pos_loss = cls_loss[..., 1:].sum() / batch_size
        cls_neg_loss = cls_loss[..., 0].sum() / batch_size
    return cls_pos_loss, cls_neg_loss

def _flat_nested_json_dict(json_dict, flatted, sep=".", start=""):
    for k, v in json_dict.items():
        if isinstance(v, dict):
            _flat_nested_json_dict(v, flatted, sep, start + sep + k)
        else:
            flatted[start + sep + k] = v

def flat_nested_json_dict(json_dict, sep=".") -> dict:
    """flat a nested json-like dict. this function make shadow copy.
    """
    flatted = {}
    for k, v in json_dict.items():
        if isinstance(v, dict):
            _flat_nested_json_dict(v, flatted, sep, k)
        else:
            flatted[k] = v
    return flatted

def example_convert_to_torch(example, dtype=torch.float32,
                             device=None) -> dict:
    device = device or torch.device("cuda:0")
    example_torch = {}
    float_names = [
        "voxels", "anchors", "reg_targets", "reg_weights", "bev_map", "rect",
        "Trv2c", "P2"
    ]

    for k, v in example.items():
        if k in float_names:
            example_torch[k] = torch.tensor(v, dtype=torch.float32, device=device).to(dtype)
        elif k in ["coordinates", "labels", "num_points"]:
            example_torch[k] = torch.tensor(
                v, dtype=torch.int32, device=device)
        elif k in ["anchors_mask"]:
            example_torch[k] = torch.tensor(
                v, dtype=torch.uint8, device=device)
        else:
            example_torch[k] = v
    return example_torch

def _predict_kitti_to_file(net,
                           example,
                           result_save_path,
                           class_names,
                           center_limit_range=None,
                           lidar_input=False):
    batch_image_shape = example['image_shape']
    batch_imgidx = example['image_idx']
    predictions_dicts = net(example)
    #print("predictions", predictions_dicts)
    # t = time.time()
    for i, preds_dict in enumerate(predictions_dicts):
        image_shape = batch_image_shape[i]
        img_idx = preds_dict["image_idx"]
        if preds_dict["bbox"] is not None or preds_dict["bbox"].size.numel():
            box_2d_preds = preds_dict["bbox"].data.cpu().numpy()
            # print("box_2d_preds", box_2d_preds)

            box_preds = preds_dict["box3d_camera"].data.cpu().numpy()
            scores = preds_dict["scores"].data.cpu().numpy()
            box_preds_lidar = preds_dict["box3d_lidar"].data.cpu().numpy()
            # write pred to file
            box_preds = box_preds[:, [0, 1, 2, 4, 5, 3,
                                      6]]  # lhw->hwl(label file format)
            label_preds = preds_dict["label_preds"].data.cpu().numpy()
            # label_preds = np.zeros([box_2d_preds.shape[0]], dtype=np.int32)
            result_lines = []
            # print("box_preds_lidar", box_preds_lidar)
            for box, box_lidar, bbox, score, label in zip(
                    box_preds, box_preds_lidar, box_2d_preds, scores,
                    label_preds):
                if not lidar_input:
                    if bbox[0] > image_shape[1] or bbox[1] > image_shape[0]:
                        continue
                    if bbox[2] < 0 or bbox[3] < 0:
                        continue
                # print(img_shape)
                if center_limit_range is not None:
                    limit_range = np.array(center_limit_range)
                    if (np.any(box_lidar[:3] < limit_range[:3])
                            or np.any(box_lidar[:3] > limit_range[3:])):
                        continue
                bbox[2:] = np.minimum(bbox[2:], image_shape[::-1])
                bbox[:2] = np.maximum(bbox[:2], [0, 0])
                result_dict = {
                    'name': class_names[int(label)],
                    'alpha': -np.arctan2(-box_lidar[1], box_lidar[0]) + box[6],
                    'bbox': bbox,
                    'location': box_lidar[:3],
                    'dimensions': box_lidar[3:6],
                    'rotation_y': box_lidar[6],
                    'score': score,
                }
                result_line = ["car", box_lidar[0], box_lidar[1], box_lidar[2],
                               box_lidar[3], box_lidar[4], box_lidar[5], box_lidar[6], score]
                # result_line = ["car", box_lidar[:3], box_lidar[3:6], box_lidar[6]]
                result_lines.append(result_line)
                # result_dict = {
                #     'name': class_names[int(label)],
                #     'alpha': -np.arctan2(-box_lidar[1], box_lidar[0]) + box[6],
                #     'bbox': bbox,
                #     'location': box[:3],
                #     'dimensions': box[3:6],
                #     'rotation_y': box[6],
                #     'score': score,
                # }
                # result_line = kitti.kitti_result_line(result_dict)
                # result_lines.append(result_line)

                # result_line = result_line.split(" ")
                # result_line = result_line[0] + " " + result_line[8] + " " \
                #               + result_line[9] + " " + result_line[10] \
                #               + " " + result_line[11] + " " + result_line[12] + " " \
                #               + " " + result_line[13] + " " + result_line[14] + " " + result_line[15]
                # print("result_line", result_line)
                # print("result_line_type", type(result_line))
                # result_lines.append(np.array(result_line))
        else:
            result_lines = []
        # result_file = f"{result_save_path}/{kitti.get_image_index_str(img_idx)}.txt"
        # # result_str = '\n'.join(result_lines)
        # result_str = np.array(result_lines)
        result_file = f"{result_save_path}/{kitti.get_image_index_str(img_idx)}.txt"
        # result_str = '\n'.join(result_lines)
        # print("result_file", result_file)
        result_lines = np.array(result_lines)
        # print("result_lines", result_lines)
        # print("result_lines_type", type(result_lines))
        results = [result_lines]
        return results
        # with open(result_file, 'w') as f:
        #     f.write(result_str)

def predict_kitti_to_anno(net,
                          example,
                          result_save_path,
                          class_names,
                          center_limit_range=None,
                          lidar_input=False,
                          global_set=None):
    print("example_shape",example.shape)
    batch_image_shape = example['image_shape']
    batch_imgidx = example['image_idx']
    predictions_dicts = net(example)
    # t = time.time()
    annos = []
    for i, preds_dict in enumerate(predictions_dicts):
        image_shape = batch_image_shape[i]
        img_idx = preds_dict["image_idvx"]
        if preds_dict["bbox"] is not None or preds_dict["bbox"].size.numel() != 0:
            box_2d_preds = preds_dict["bbox"].detach().cpu().numpy()
            box_preds = preds_dict["box3d_camera"].detach().cpu().numpy()
            scores = preds_dict["scores"].detach().cpu().numpy()
            box_preds_lidar = preds_dict["box3d_lidar"].detach().cpu().numpy()
            # write pred to file
            label_preds = preds_dict["label_preds"].detach().cpu().numpy()
            # label_preds = np.zeros([box_2d_preds.shape[0]], dtype=np.int32)
            anno = kitti.get_start_result_anno()
            num_example = 0
            for box, box_lidar, bbox, score, label in zip(
                    box_preds, box_preds_lidar, box_2d_preds, scores,
                    label_preds):
                if not lidar_input:
                    if bbox[0] > image_shape[1] or bbox[1] > image_shape[0]:
                        continue
                    if bbox[2] < 0 or bbox[3] < 0:
                        continue
                # print(img_shape)
                if center_limit_range is not None:
                    limit_range = np.array(center_limit_range)
                    if (np.any(box_lidar[:3] < limit_range[:3])
                            or np.any(box_lidar[:3] > limit_range[3:])):
                        continue
                bbox[2:] = np.minimum(bbox[2:], image_shape[::-1])
                bbox[:2] = np.maximum(bbox[:2], [0, 0])
                anno["name"].append(class_names[int(label)])
                anno["truncated"].append(0.0)
                anno["occluded"].append(0)
                anno["alpha"].append(-np.arctan2(-box_lidar[1], box_lidar[0]) +
                                     box[6])
                anno["bbox"].append(bbox)
                anno["dimensions"].append(box[3:6])
                anno["location"].append(box[:3])
                anno["rotation_y"].append(box[6])
                if global_set is not None:
                    for i in range(100000):
                        if score in global_set:
                            score -= 1 / 100000
                        else:
                            global_set.add(score)
                            break
                anno["score"].append(score)

                num_example += 1
            if num_example != 0:
                anno = {n: np.stack(v) for n, v in anno.items()}
                annos.append(anno)
            else:
                annos.append(kitti.empty_result_anno())
        else:
            annos.append(kitti.empty_result_anno())
        num_example = annos[-1]["name"].shape[0]
        annos[-1]["image_idx"] = np.array(
            [img_idx] * num_example, dtype=np.int64)
    return annos


def evaluate(config_path,
             model_dir,
             result_path=None,
             predict_test=False,
             ckpt_path=None,
             measure_time=None,
             batch_size=1
             ):
    if predict_test:
        result_name = 'predict_test'
    else:
        result_name = 'eval_results'
    model_dir = pathlib.Path(model_dir)
    if result_path is None:
        result_path = model_dir / result_name
    else:
        result_path = pathlib.Path(result_path)
    config = pipeline_pb2.TrainEvalPipelineConfig()
    with open(config_path, "r") as f:
        proto_str = f.read()
        text_format.Merge(proto_str, config)

    input_cfg = config.eval_input_reader
    model_cfg = config.model.second
    train_cfg = config.train_config

    center_limit_range = model_cfg.post_center_limit_range
    ######################
    # BUILD VOXEL GENERATOR
    ######################
    voxel_generator = voxel_builder.build(model_cfg.voxel_generator)
    bv_range = voxel_generator.point_cloud_range[[0, 1, 3, 4]]
    box_coder = box_coder_builder.build(model_cfg.box_coder)
    target_assigner_cfg = model_cfg.target_assigner
    target_assigner = target_assigner_builder.build(target_assigner_cfg,
                                                    bv_range, box_coder)
    net = second_builder.build(model_cfg, voxel_generator, target_assigner, measure_time=measure_time)
    net.cuda()
    if ckpt_path is None:
        torchplus.train.try_restore_latest_checkpoints(model_dir, [net])
    else:
        torchplus.train.restore(ckpt_path, net)
    if train_cfg.enable_mixed_precision:
        net.half()
        print("half inference!")
        net.metrics_to_float()
        net.convert_norm_to_float(net)

    net.eval()

    result_path_step = result_path / f"step_{net.get_global_step()}"
    result_path_step.mkdir(parents=True, exist_ok=True)

    return net,voxel_generator,target_assigner,result_path_step,model_cfg,center_limit_range

def test(points,
         net,
         voxel_generator,
         target_assigner,
         result_path_step,
         model_cfg,
         center_limit_range):
    # net,voxel_generator,target_assigner,result_path_step\
    #     ,model_cfg,center_limit_range=evaluate(config_path=config_path,
    #                                            model_dir=model_dir,
    #                                            batch_size=1)
    #print(points.shape)

    start = time.clock()
    example=prep_pointcloud(kitti_points=points,
                            root_path=None,
                            voxel_generator=voxel_generator,
                            target_assigner=target_assigner,
                            db_sampler=None,
                            max_voxels=40000,
                            class_names=['Car'],
                            remove_outside_points=False,
                            training=False,
                            create_targets=True,
                            shuffle_points=False,
                            reduce_valid_area=False,
                            remove_unknown=False,
                            gt_rotation_noise=[-np.pi / 3, np.pi / 3],
                            gt_loc_noise_std=[1.0, 1.0, 1.0],
                            global_rotation_noise=[-np.pi / 4, np.pi / 4],
                            global_scaling_noise=[0.95, 1.05],
                            global_random_rot_range=[0.78, 2.35],
                            generate_bev=False,
                            without_reflectivity=False,
                            num_point_features=4,
                            anchor_area_threshold=-1,
                            gt_points_drop=0.0,
                            gt_drop_max_keep=0,
                            remove_points_after_sample=False,
                            anchor_cache=None,
                            remove_environment=False,
                            random_crop=False,
                            reference_detections=None,
                            add_rgb_to_points=False,
                            lidar_input=False,
                            unlabeled_db_sampler=None,
                            out_size_factor=8,
                            min_gt_point_dict=None,
                            bev_only=False,
                            use_group_id=False,
                            out_dtype=np.float32
                            )
    # eval_dataloader = torch.utils.data.DataLoader(
    #     example,
    #     batch_size=1,
    #     shuffle=False,
    #     num_workers=3,  # input_cfg.num_workers,
    #     pin_memory=False,
    #     collate_fn=merge_second_batch)

    #example=iter(eval_dataloader)
    #print("coordinates:",example["coordinates"][1,1])
    coordinates_shape =example["coordinates"].shape[0]
    #print("coor:",coordinates_shape)
    a= np.zeros([coordinates_shape,1],int)
    example["coordinates"]=np.c_[a,example["coordinates"]]
    #print("coordinates:", example["coordinates"][1, 2])
    example["rect"]=example["rect"].reshape([1,4,4])
    example["Trv2c"]=example["Trv2c"].reshape([1,4,4])
    example["P2"]=example["P2"].reshape([1,4,4])
    anchors_shape = example["anchors"].shape[0]
    example["anchors"]=example["anchors"].reshape([1,anchors_shape,7])
    example["image_idx"] = np.array([random.randint(1, 100000)])
    #print(example["image_idx"])
    example["image_shape"] = np.array([375,1242]).reshape([1,2])
    example = example_convert_to_torch(example, torch.float32)
    class_names = target_assigner.classes
    #print("coordinates:", example["coordinates"].shape[0])
    #result=net(example)
    result = _predict_kitti_to_file(net,
                                    example,
                                    result_path_step,
                                    class_names,
                                    center_limit_range,
                                    model_cfg.lidar_input
                                    )
    #print(result["bbox"][1,1])
    elapsed = (time.clock() - start)
    #print("Time used:", elapsed)
    #print("result:",result)
    return result

if __name__ == '__main__':
    # points = np.load("/root/second-1.5/points.npy")
    # net,voxel_generator,target_assigner,result_path_step\
    #     ,model_cfg,center_limit_range=evaluate(config_path="/root/second-1.5/second/configs/car.fhd.config",
    #                                            model_dir="/root/model/218/",
    #                                            batch_size=1)
    # test(points=points,
    #      net=net,
    #      voxel_generator=voxel_generator,
    #      target_assigner=target_assigner,
    #      result_path_step=result_path_step,
    #      model_cfg=model_cfg,
    #      center_limit_range=center_limit_range
    #      )
    fire.Fire()


