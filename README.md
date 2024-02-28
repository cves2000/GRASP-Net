 # GRASP-Net: Geometric Residual Analysis and Synthesis for Point Cloud Compression
Created by Jiahao Pang, Muhammad Asad Lodhi and Dong Tian from <a href="https://www.interdigital.com/" target="_blank">InterDigital</a>.

<p align="center">
    <img src="assets/framework.png" alt="framework" width="60%" />
</p>

## Introduction

This repository contains the implementation of the [GRASP-Net](https://arxiv.org/pdf/2209.04401.pdf) paper accepted in the ACM MM 2022 Workshop on APCCPA. 3D point clouds are locally sparse, which makes it difficult to capture the correlation among neighboring points for point cloud compression (PCC). To address this issue, we proposed GRASP-Net, a heterogeneous framework for AI-based PCC, which combines both convolutional layers and native point-based networks to effectively compress the geometry of an input point cloud in a lossy manner. Experimentation on both dense and sparse point clouds demonstrate the state-of-the-art compression performance achieved by our proposal. Our GRASP-Net is implemented based on the [PccAI](https://github.com/InterDigitalInc/PccAI) (*pick-kai*) framework—a PyTorch-based framework for conducting AI-based PCC experiments.

## Installation

We tested our implementation under two different virtual environments with conda:
* Python 3.6, PyTorch 1.7.0, and CUDA 10.1. For this configuration, please launch the installation script `install_torch-1.7.0+cu-10.1.sh` with the following command:
```bash
echo y | conda create -n grasp python=3.6 && conda activate grasp && ./install_torch-1.7.0+cu-10.1.sh
```
* Python 3.8, PyTorch 1.8.1, and CUDA 11.1. For this configuration, please launch the installation script `install_torch-1.8.1+cu-11.2.sh` with the following command:
```bash
echo y | conda create -n grasp python=3.8 && conda activate grasp && ./install_torch-1.8.1+cu-11.2.sh
```
It is *highly recommended* to check the installation scripts which describe the details of the necessary packages. Note that [torchac](https://github.com/fab-jul/torchac) is used for arithmetic coding and [plyfile](https://github.com/dranjan/python-plyfile) is used for the reading/writing of PLY files. These two packages are under GPL license. By replacing them with another library providing the same functionality, our implementation can still run.

After that, put the binary of [`tmc3`](https://github.com/MPEGGroup/mpeg-pcc-tmc13) (MPEG G-PCC) and `pc_error` (D1 & D2 computation tool used in the MPEG group) under the `third_party` folder. A publicly-available version of `pc_error` can be found [here](https://github.com/NJUVISION/PCGCv2/blob/master/pc_error_d). To use it for the benchmarking of GRASP-Net, please download and rename it to `pc_error`.

环境配置：GRASP-Net 的实现在两种不同的虚拟环境下进行了测试，这两种环境都使用了 conda。具体的安装步骤包括创建一个名为 grasp 的 conda 环境，然后在这个环境中安装特定版本的 Python、PyTorch 和 CUDA。安装脚本 install_torch-1.7.0+cu-10.1.sh 和 install_torch-1.8.1+cu-11.2.sh 包含了安装过程的详细信息。
必要的包：GRASP-Net 使用了 torchac 和 plyfile 这两个包。torchac 用于算术编码，plyfile 用于读写 PLY 文件。这两个包都是 GPL 许可的。如果你想使用其他提供相同功能的库，只需将这两个库替换掉，GRASP-Net 的实现仍然可以运行。
第三方工具：在 third_party 文件夹下放置 tmc3（MPEG G-PCC）和 pc_error（在 MPEG 组中用于计算 D1 和 D2 的工具）的二进制文件。pc_error 的公开版本可以在这里找到。为了使用它对 GRASP-Net 进行基准测试，请下载并将其重命名为 pc_error。
总的来说，这段话是关于如何安装和设置 GRASP-Net 的说明。
## Datasets
Create a `datasets` folder then put all the datasets below. One may create soft links to the existing datasets to save space.

数据集：GRASP-Net 使用 Ford 序列和 ModelNet40 数据集进行训练。Ford 序列的第一部分用于训练，其余部分用于基准测试。ModelNet40 数据集用于表面点云的训练。数据集应该被放置在 datasets 文件夹下。
### Ford Sequences

The GRASP-Net uses the first *Ford* sequences for training and the other two sequences for benchmarking. They are arranged as follows:
```bash
${ROOT_OF_THE_REPO}/datasets/ford
                               ├── ford_01_q1mm
                               ├── ford_02_q1mm
                               └── ford_03_q1mm
                                       ├── Ford_03_vox1mm-0200.ply
                                       ├── Ford_03_vox1mm-0201.ply
                                       ├── Ford_03_vox1mm-0202.ply
                                       ...
                                       └── Ford_03_vox1mm-1699.ply
```
这段话是关于如何组织 Ford 序列数据集的说明。Ford 序列是一种点云数据，用于训练和基准测试 GRASP-Net。
在这个例子中，数据集被组织在 ${ROOT_OF_THE_REPO}/datasets/ford 文件夹下，其中 ${ROOT_OF_THE_REPO} 是代码仓库的根目录。这个文件夹下有三个子文件夹 ford_01_q1mm、ford_02_q1mm 和 ford_03_q1mm，它们分别包含了不同的 Ford 序列。
GRASP-Net 使用第一个 Ford 序列（即 ford_01_q1mm）进行训练，使用其他两个序列（即 ford_02_q1mm 和 ford_03_q1mm）进行基准测试。
每个 Ford 序列文件夹下都有多个 .ply 文件，每个文件都是一个点云数据。例如，ford_03_q1mm 文件夹下就有 Ford_03_vox1mm-0200.ply、Ford_03_vox1mm-0201.ply、Ford_03_vox1mm-0202.ply 等多个文件。

### ModelNet40

The GRASP-Net uses ModelNet40 to train for the case of surface point clouds. Our ModelNet40 data loader is built on top of the loader of PyTorch Geometric. For the first run, it will automatically download the ModelNet40 data under the `datasets` folder and preprocess it. 

这段话是关于如何使用 ModelNet40 数据集进行训练的说明。以下是对这段话的详细解释：
ModelNet40：ModelNet40 是一个常用的 3D 形状分类数据集，包含了 40 个类别的 3D 模型。在这个例子中，GRASP-Net 使用 ModelNet40 数据集来训练处理表面点云的模型。
数据加载器：GRASP-Net 的 ModelNet40 数据加载器是基于 PyTorch Geometric 的加载器构建的。PyTorch Geometric 是一个用于图形结构数据的 PyTorch 库，提供了一种方便的方式来加载和处理图形数据。
自动下载和预处理：在第一次运行时，数据加载器会自动下载 ModelNet40 数据，并将其保存在 datasets 文件夹下。然后，它会对数据进行预处理，以便于后续的训练使用。
### Surface Point Clouds

The test set of the surface point clouds should be organized as shown below. Note that the point clouds are selected according to the MPEG recommendation [w21696](https://www.mpeg.org/wp-content/uploads/mpeg_meetings/139_OnLine/w21696.zip).
```bash
${ROOT_OF_THE_REPO}/datasets/cat1
                               ├──A
                               │  ├── soldier_viewdep_vox12.ply
                               │  ├── boxer_viewdep_vox12.ply
                               │  ├── Facade_00009_vox12.ply
                               │  ├── House_without_roof_00057_vox12.ply
                               │  ├── queen_0200.ply
                               │  ├── soldier_vox10_0690.ply
                               │  ├── Facade_00064_vox11.ply
                               │  ├── dancer_vox11_00000001.ply
                               │  ├── Thaidancer_viewdep_vox12.ply
                               │  ├── Shiva_00035_vox12.ply
                               │  ├── Egyptian_mask_vox12.ply
                               │  └── ULB_Unicorn_vox13.ply
                               └──B
                                  ├── Arco_Valentino_Dense_vox12.ply
                                  └── Staue_Klimt_vox12.ply
```
Note that the file names are case-sensitive. Users may also put other surface point clouds in the `cat1/A` or `cat1/B` folders for additional testing.

在这个例子中，点云数据集被组织在 ${ROOT_OF_THE_REPO}/datasets/cat1 文件夹下，其中 ${ROOT_OF_THE_REPO} 是代码仓库的根目录。这个文件夹下有两个子文件夹 A 和 B，它们分别包含了多个 .ply 文件，每个文件都是一个点云数据。
这些点云数据是根据 MPEG 的推荐 w21696 选择的。MPEG 是一种用于编码数字音频和视频的标准。
请注意，文件名是区分大小写的。这意味着 soldier_viewdep_vox12.ply 和 Soldier_viewdep_vox12.ply 会被视为两个不同的文件。
此外，用户也可以将其他的点云数据放入 cat1/A 或 cat1/B 文件夹中进行额外的测试。
总的来说，这段话是关于如何组织和使用点云数据集的说明。
## Basic Usages

The core of the training and benchmarking code are put below the `pccai/pipelines` folder. They are called by their wrappers below the `experiments` folder. The basic way to launch experiments with PccAI is:
 ```bash
 ./scripts/run.sh ./scripts/[filename].sh [launcher] [GPU ID(s)]
 ```
where `launcher` can be `s` (slurm), `d` (direct, run in background) and `f` (direct, run in foreground). `GPU ID(s)` can be ignored when launched with slurm. The results (checkpoints, point cloud files, log, *etc.*) will be generated under the `results/[filename]` folder. Note that multi-GPU training/benchmarking is not supported by GRASP-Net.

1.pccai/pipelines 文件夹下存放的是训练和基准测试代码的核心部分。
2.这些核心代码会被 experiments 文件夹下的包装器（wrapper）调用。
3.启动 PccAI 实验的基本方式是运行以下命令：
./scripts/run.sh ./scripts/[filename].sh [launcher] [GPU ID(s)]
其中：
./scripts/run.sh 是运行脚本的命令。
./scripts/[filename].sh 是你要运行的脚本文件，你需要将 [filename] 替换为实际的文件名。
[launcher] 是启动器类型，可以是 s（表示 slurm），d（表示直接运行，并在后台运行），f（表示直接运行，并在前台运行）。
[GPU ID(s)] 是你要使用的 GPU 的 ID，如果你使用的是 slurm 启动器，可以忽略这个参数。
4.运行上述命令后，实验的结果（包括检查点、点云文件、日志等）将会生成在 results/[filename] 文件夹下。
5.请注意，GRASP-Net 不支持多 GPU 训练/基准测试。
总的来说，这段话是关于如何使用 PccAI 进行实验的基本说明。

### Training

Take the training on the Ford sequences as an example, one can directly run
 ```bash
./scripts/run.sh ./scripts/train_grasp/train_lidar_ford/train_lidar_ford_r01.sh d 0
 ```
which trains the model of the first rate point when operating on the Ford sequences. The trained model will be generated under the `results/train_lidar_ford_r01` folder. Note that all the models for the five rate points should be trained to have the complete R-D curves. Please follow the same way to train the models for other datasets. All the training scripts are provided under the `scripts/train_grasp` folder.

To understand the meanings of the options in the scripts for benchmarking/training, refer to `pccai/utils/option_handler.py` for details.

这段话是关于如何训练一个模型的说明。这里以 Ford 序列为例，进行了一次模型训练的演示。具体步骤如下：
运行命令 ./scripts/run.sh ./scripts/train_grasp/train_lidar_ford/train_lidar_ford_r01.sh d 0。这个命令会启动一个训练脚本，该脚本会训练一个模型，该模型在操作 Ford 序列时对应于第一个速率点。
训练完成后，训练好的模型将会被保存在 results/train_lidar_ford_r01 文件夹下。
请注意，为了得到完整的 R-D 曲线，你需要训练所有五个速率点的模型。R-D 曲线是一种用于评估压缩算法性能的工具，它展示了压缩率（R）和失真度（D）之间的关系。
你可以按照同样的方式训练其他数据集的模型。所有的训练脚本都可以在 scripts/train_grasp 文件夹下找到。
如果你想了解脚本中选项的含义，可以参考 pccai/utils/option_handler.py 文件。这个文件包含了处理选项的各种函数和类，可以帮助你理解和使用这些选项。
 ### Benchmarking

The trained models of GRASP-Net are released [here](https://www.dropbox.com/s/80z19597tcpfdqn/grasp-net_models_20220919.zip?dl=0). Please put the downloaded folders `grasp_surface_solid`, `grasp_surface_dense`, `grasp_surface_sparse`, and `grasp_lidar_ford` right beneath the `results` folder. Then to benchmark the performance on the Ford sequences, one can directly run
 ```bash
./scripts/run.sh ./scripts/bench_grasp/bench_lidar_ford/bench_lidar_ford_all.sh d 0
 ```
which benchmarks all the rate points on GPU #0 and generates the statistics for each rate point in the CSV file `results/bench_lidar_ford_all/mpeg_report.csv`.

Alternatively, one can use the following command lines for benchmarking the five rate points individually, followed by merging the generated CSV files:
 ```bash
for i in {1..5}
do
   ./scripts/run.sh ./scripts/bench_grasp/bench_lidar_ford/bench_lidar_ford_r0$i.sh f 0
done
python ./utils/merge_csv.py --input_files ./results/bench_lidar_ford_r01/mpeg_report.csv ./results/bench_lidar_ford_r02/mpeg_report.csv ./results/bench_lidar_ford_r03/mpeg_report.csv ./results/bench_lidar_ford_r04/mpeg_report.csv ./results/bench_lidar_ford_r05/mpeg_report.csv --output_file ./results/grasp_lidar_ford/mpeg_report.csv
 ```
All the benchmarking scripts for different categories of point clouds are provided under the `scripts/bench_grasp` folder.

BD metrics and R-D curves are generated via the *AI-PCC-Reporting-Template* with [commit 01a6857](https://github.com/yydlmzyz/AI-PCC-Reporting-Template/tree/01a68579f04b4741de77b193f168d730456cf0d6). For example, run the following command right beneath its repository:
```bash
python test.py --csvdir1='csvfiles/reporting_template_lossy.csv' --csvdir2='/PATH/TO/mpeg_report.csv' --csvdir_stats='csvfiles/reporting_template_stats.csv' --xlabel='bppGeo' --ylabel='d1T'
```
It can also generate the average results for a certain point cloud category:
```bash
python test_mean.py --category='am_frame' --csvdir1='csvfiles/reporting_template_lossy.csv' --csvdir2='/PATH/TO/mpeg_report.csv' --csvdir_stats='csvfiles/reporting_template_stats.csv' --xlabel='bppGeo' --ylabel='d1T'
```

Replace `d1T` with `d2T` for computing the D2 metrics. The benchmarking of surface point clouds can be done in the same way. Example R-D curves and the results of all rate points of GRASP-Net are placed under the `assets` folder.

这段话是关于如何使用 GRASP-Net 进行基准测试的说明。以下是对这段话的详细解释：
模型：GRASP-Net 的训练模型可以在给出的链接中下载。下载后的文件夹 grasp_surface_solid、grasp_surface_dense、grasp_surface_sparse 和 grasp_lidar_ford 应该被放置在 results 文件夹下。
基准测试：你可以运行给出的命令来对 Ford 序列进行基准测试。这个命令会在 GPU #0 上对所有的速率点进行基准测试，并在 CSV 文件 results/bench_lidar_ford_all/mpeg_report.csv 中生成每个速率点的统计数据。
替代方法：你也可以使用给出的命令行来分别对五个速率点进行基准测试，然后合并生成的 CSV 文件。
脚本：所有的基准测试脚本都提供在 scripts/bench_grasp 文件夹下。
度量和曲线：BD 度量和 R-D 曲线是通过 AI-PCC-Reporting-Template 生成的。你可以运行给出的命令来生成这些度量和曲线。这个命令会在其仓库下运行。
平均结果：你也可以生成某个点云类别的平均结果。
替换：你可以将 d1T 替换为 d2T 来计算 D2 度量。表面点云的基准测试可以以相同的方式进行。
示例：GRASP-Net 的所有速率点的示例 R-D 曲线和结果都放置在 assets 文件夹下。
## Cite This Work
Please cite our work if you find it useful for your research:
```
@article{pang2022grasp,
  title={GRASP-Net: Geometric Residual Analysis and Synthesis for Point Cloud Compression},
  author={Pang, Jiahao and Lodhi, Muhammad Asad and Tian, Dong},
  bookitle={ACM MM Workshop on APCCPA},
  year={2022}
}
```
## License
GRASP-Net is released under the BSD License, see `LICENSE` for details.

## Contacts
Please contact Jiahao Pang (jiahao.pang@interdigital.com), the main contributor of both GRASP-Net and PccAI, if you have any questions.

## Related Resources
 * [PccAI](https://github.com/InterDigitalInc/PccAI)
 * [3D Point Capsule Networks](https://github.com/yongheng1991/3D-point-capsule-networks)
 * [MinkowskiEngine](https://github.com/NVIDIA/MinkowskiEngine)
 * [PCGCv2](https://github.com/NJUVISION/PCGCv2/)
 * [AI-PCC-Reporting-Template](https://github.com/yydlmzyz/AI-PCC-Reporting-Template)
 * [TMC13](https://github.com/MPEGGroup/mpeg-pcc-tmc13)

<p align="center">
    <img src="assets/demo.png" alt="demo" width="90%" /> 
</p>
