#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved


import os
import cv2
import logging
from collections import OrderedDict

import detectron2.utils.comm as comm
from detectron2.utils.visualizer import Visualizer
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets.coco import load_coco_json
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, launch
from detectron2.engine import HookBase
from detectron2.evaluation import COCOEvaluator, PascalVOCDetectionEvaluator, verify_results
from detectron2.modeling.roi_heads import ROI_HEADS_REGISTRY, Res5ROIHeads
from detectron2.modeling import GeneralizedRCNNWithTTA
from detectron2.layers import get_norm
from detectron2.data.datasets import register_coco_instances


# ROOTS
DATASET_ROOT = '../../benchmarks/detection/datasets/lus'
ANN_ROOT = os.path.join(DATASET_ROOT,'annotations')
TRAIN_PATH = os.path.join(DATASET_ROOT,'lus_train')
VAL_PATH = os.path.join(DATASET_ROOT,'lus_val')
TRAIN_JSON = os.path.join(ANN_ROOT,'instances_trainlus.json')
VAL_JSON = os.path.join(ANN_ROOT,'instances_vallus.json')

register_coco_instances("lus_train", {}, TRAIN_JSON, TRAIN_PATH)
register_coco_instances("lus_val", {}, VAL_JSON, VAL_PATH)



@ROI_HEADS_REGISTRY.register()
class Res5ROIHeadsExtraNorm(Res5ROIHeads):
    """
    As described in the MOCO paper, there is an extra BN layer
    following the res5 stage.
    """

    def _build_res5_block(self, cfg):
        seq, out_channels = super()._build_res5_block(cfg)
        norm = cfg.MODEL.RESNETS.NORM
        norm = get_norm(norm, out_channels)
        seq.add_module("norm", norm)
        return seq, out_channels


class Trainer(DefaultTrainer):

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return COCOEvaluator(dataset_name, cfg, True, output_folder)

class BestCheckpointer(HookBase):
    def __init__(self):
        super().__init__()

    def after_step(self):
        # No way to use **kwargs

        ##ONly do this analys when trainer.iter is divisle by checkpoint_epochs
        curr_val = self.trainer.storage.latest().get('bbox/AP50', 0)

        import math
        if type(curr_val) != int:
            curr_val = curr_val[0]
            if math.isnan(curr_val):
                curr_val = 0

        try:
            _ = self.trainer.storage.history('max_bbox/AP50')
        except:
            self.trainer.storage.put_scalar('max_bbox/AP50', curr_val)

        max_val = self.trainer.storage.history('max_bbox/AP50')._data[-1][0]

        #print(curr_val, max_val)
        if curr_val > max_val:
            print("\n%s > %s  save!!\n"%(curr_val,max_val))
            self.trainer.storage.put_scalar('max_bbox/AP50', curr_val)
            self.trainer.checkpointer.save("model_best")
            #self.step(self.trainer.iter)




def setup(args):
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)

    cfg.DATASETS.TRAIN = ("lus_train",)
    cfg.DATASETS.TEST = ("lus_val",)


    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):
    cfg = setup(args)

    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(
            model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
                cfg.MODEL.WEIGHTS, resume=args.resume)
        res = Trainer.test(cfg, model)
        return res

    trainer = Trainer(cfg)
    trainer.register_hooks([BestCheckpointer()])
    trainer.resume_or_load(resume=args.resume)
    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args, ),
    )
