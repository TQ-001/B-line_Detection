
## Transferring to Detection

We follow the evaluation setting in MoCo when trasferring to object detection.

### Instruction

1. Install [detectron2](https://github.com/facebookresearch/detectron2/blob/master/INSTALL.md).

2. Put dataset under "benchmarks/detection/datasets" directory,
   following the [directory structure](https://github.com/facebookresearch/detectron2/tree/master/datasets)
	 requried by detectron2.

3. Put weight file under "benchmarks/detection/pth" directory, 
   using "convert-pretrain-to-detectron2.py" to convert .pth to .pkl, 
   and put .pkl files under "benchmarks/detection/pkl"
