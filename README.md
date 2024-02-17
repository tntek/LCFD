# LCFD
Code (pytorch) for ['Unified Source-Free Domain Adaptation']() on Office-Home, VisDA-C.

### Preliminary

You need to download the, [Office-Home](https://drive.google.com/file/d/0B81rNlvomiwed0V1YUxQdC1uOTg/view), [VisDA-C](https://github.com/VisionLearningGroup/taskcv-2017-public/tree/master/classification) dataset,modify the path of images in each '.txt' under the folder './data/'.

The experiments are conducted on one GPU (NVIDIA RTX TITAN).

- python == 3.7.11
- pytorch ==1.13.0
- torchvision == 0.14.0
- numpy, scipy, sklearn, PIL, argparse, tqdm

## Prepare pretrain model for source model
