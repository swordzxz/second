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
    # print("predictions", predictions_dicts)
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


def evaluate(points,
             config_path,
             model_dir,
             result_path=None,
             predict_test=False,
             ckpt_path=None,
             ref_detfile=None,
             pickle_result=True,
             measure_time=False,
             batch_size=None):
    model_dir = pathlib.Path(model_dir)
    if predict_test:
        result_name = 'predict_test'
    else:
        result_name = 'eval_results'
    if result_path is None:
        result_path = model_dir / result_name
    else:
        result_path = pathlib.Path(result_path)
    config = pipeline_pb2.TrainEvalPipelineConfig()
    # print("config", config)
    with open(config_path, "r") as f:
        proto_str = f.read()
        # print("proto_str", proto_str)
        text_format.Merge(proto_str, config)
        # after text_format.Merge(proto_str, config), "proto_str" and "config" are same.
        # print("config", config)
        # print("proto_str", proto_str)
        # print("after_config")
    input_cfg = config.eval_input_reader
    model_cfg = config.model.second
    train_cfg = config.train_config
    # print("train_cfg", train_cfg)

    center_limit_range = model_cfg.post_center_limit_range
    # print("center_limit_range", center_limit_range)
    # center_limit_range[0.0, -40.0, -3.0, 70.4000015258789, 40.0, 0.0]

    ######################
    # BUILD VOXEL GENERATOR
    ######################
    voxel_generator = voxel_builder.build(model_cfg.voxel_generator)
    print("voxel_generator_type", type(voxel_generator))
    bv_range = voxel_generator.point_cloud_range[[0, 1, 3, 4]]
    box_coder = box_coder_builder.build(model_cfg.box_coder)
    print("box_coder_builder_type", type(box_coder_builder))
    target_assigner_cfg = model_cfg.target_assigner
    target_assigner = target_assigner_builder.build(target_assigner_cfg,
                                                    bv_range, box_coder)
    class_names = target_assigner.classes
    # print("class_name", class_names)
    # print("class_name_type", type(class_names))

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
    batch_size = batch_size or input_cfg.batch_size

    # raw_lidar = np.fromfile("/home/ubuntu/dataset/kitti_datasets/training/velodyne/000000.bin", dtype=np.float32).reshape((-1, 4))
    raw_lidar = points
    print("raw_lidar_shape", raw_lidar.shape)
    # print("raw_lidar_type", type(raw_lidar))

    eval_dataset = input_reader_builder.build(
        raw_lidar,
        input_cfg,
        model_cfg,
        training=False,
        voxel_generator=voxel_generator,
        target_assigner=target_assigner)
    # print("eval_dataset_type", type(eval_dataset))
    # print("eval_dataset", eval_dataset)
    eval_dataloader = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,  # input_cfg.num_workers,
        pin_memory=False,
        collate_fn=merge_second_batch)
    # print("eval_dataloader_type", type(eval_dataloader))
    # print("eval_dataloader", eval_dataloader)

    if train_cfg.enable_mixed_precision:
        float_dtype = torch.float16
    else:
        float_dtype = torch.float32

    net.eval()
    result_path_step = result_path / f"step_{net.get_global_step()}"
    result_path_step.mkdir(parents=True, exist_ok=True)
    t = time.time()
    dt_annos = []
    global_set = None
    print("Generate output labels...")
    bar = ProgressBar()
    bar.start((len(eval_dataset) + batch_size - 1) // batch_size)
    prep_example_times = []
    prep_times = []
    t2 = time.time()
    print("before iter")
    i = 0
    for example in iter(eval_dataloader):
        i = i + 1
        if i > 1:
            break
        print("example keys", example.keys())
        print("example_num_points", example['num_points'])
        if measure_time:
            prep_times.append(time.time() - t2)
            t1 = time.time()
            torch.cuda.synchronize()
        example = example_convert_to_torch(example, float_dtype)
        if measure_time:
            torch.cuda.synchronize()
            prep_example_times.append(time.time() - t1)

        result = _predict_kitti_to_file(net, example, result_path_step, class_names,
                                        center_limit_range, model_cfg.lidar_input)
        # print("result", result)
        # if pickle_result:
        #     dt_annos += predict_kitti_to_anno(
        #         net, example, class_names, center_limit_range,
        #         model_cfg.lidar_input, global_set)
        # else:
        #     _predict_kitti_to_file(net, example, result_path_step, class_names,
        #                            center_limit_range, model_cfg.lidar_input)
        # print(json.dumps(net.middle_feature_extractor.middle_conv.sparity_dict))
        bar.print_bar()
        if measure_time:
            t2 = time.time()

        return result

    # sec_per_example = len(eval_dataset) / (time.time() - t)
    # print(f'generate label finished({sec_per_example:.2f}/s). start eval:')
    # if measure_time:
    #     print(f"avg example to torch time: {np.mean(prep_example_times) * 1000:.3f} ms")
    #     print(f"avg prep time: {np.mean(prep_times) * 1000:.3f} ms")
    # for name, val in net.get_avg_time_dict().items():
    #     print(f"avg {name} time = {val * 1000:.3f} ms")
    # if not predict_test:
    #     gt_annos = [info["annos"] for info in eval_dataset.dataset.kitti_infos]
    #     if not pickle_result:
    #         dt_annos = kitti.get_label_annos(result_path_step)
    #     result = get_official_eval_result(gt_annos, dt_annos, class_names)
    #     # print(json.dumps(result, indent=2))
    #     print(result)
    #     result = get_coco_eval_result(gt_annos, dt_annos, class_names)
    #     print(result)
    #     if pickle_result:
    #         with open(result_path_step / "result.pkl", 'wb') as f:
    #             pickle.dump(dt_annos, f)


def save_config(config_path, save_path):
    config = pipeline_pb2.TrainEvalPipelineConfig()
    with open(config_path, "r") as f:
        proto_str = f.read()
        text_format.Merge(proto_str, config)
    ret = text_format.MessageToString(config, indent=2)
    with open(save_path, 'w') as f:
        f.write(ret)


if __name__ == '__main__':
    points = np.load("/root/second-1.5/points.npy")
    start = time.clock()
    results = evaluate(points=points,
                       config_path="/root/second-1.5/second/configs/car.fhd.config", model_dir="/root/model/218/", batch_size=1)
    elapsed = (time.clock() - start)
    print("Time used:", elapsed)

    fire.Fire()
