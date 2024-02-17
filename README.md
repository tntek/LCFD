# LCFD
Code (pytorch) for ['Unified Source-Free Domain Adaptation']() on Office-Home, VisDA-C.

### Preliminary

You need to download the, [Office-Home](https://drive.google.com/file/d/0B81rNlvomiwed0V1YUxQdC1uOTg/view), [VisDA-C](https://github.com/VisionLearningGroup/taskcv-2017-public/tree/master/classification) dataset,modify the path of images in each '.txt' under the folder './data/'.In addition, class name files for each dataset also under the folder './data/'.The prepared directory would look like:

```bash
├── data
    ├── office-home
        ├── Art_list.txt
        ├── classname.txt
        ├── Clipart_list.txt
        ├── Product_list.txt
        ├── RealWorld_list.txt
    ├── VISDA-C
        ├── classname.txt
        ├── train_list.txt
        ├── validation_list.txt
```
The experiments are conducted on one GPU (NVIDIA RTX TITAN).

- python == 3.7.11
- pytorch ==1.13.0
- torchvision == 0.14.0
- numpy, scipy, sklearn, PIL, argparse, tqdm

we also provide a conda environment.

```bash
conda update conda
conda env create -f environment.yml
conda activate sfa 
```

## Training
We provide config files for experiments. Before the source-free domain adaptation, we should have the source model. You may train the source model like:

```bash
CUDA_VISIBLE_DEVICES=0 python image_target_oh_vs.py --cfg "cfgs/[office-home/visda]/source.yaml" SETTING.S [0,1,2] SETTING.T [0,1,2]
```
We also provide the pre-trained source models which can be [downloaded from here](https://drive.google.com/drive/folders/1NyFXBpeqjYU45DaXeIpNOzP4tjy2UF5T?usp=drive_link).

After obtaining the source models, put them under `${CKPT_DIR}` to and run :

```bash
CUDA_VISIBLE_DEVICES=0 python image_target_oh_vs.py --cfg "cfgs/[office-home/visda]/lcfd.yaml" SETTING.S [0,1,2] SETTING.T [0,1,2]
```
to execute source-free domain adaptation.

### Acknowledgement


The code is based on [SHOT (ICML 2020, also source-free)](https://github.com/tim-learn/SHOT), [IIC](https://github.com/sebastiani/IIC) and [CoOp](https://github.com/KaiyangZhou/CoOp).


### Contact

- tntechlab@hotmail.com
