# 10708 S23 Course Project


## Dataset
Our dataset download process follows the [ReferIt3D benchmark](https://github.com/referit3d/referit3d).

Specifically, you will need to
- (1) Download `sr3d_train.csv` and `sr3d_test.csv` from this [link](https://drive.google.com/drive/folders/1DS4uQq7fCmbJHeE-rEbO8G1-XatGEqNV)
- (2) Download scans from ScanNet and process them according to this [link](https://github.com/referit3d/referit3d/blob/eccv/referit3d/data/scannet/README.md). This should result in a `keep_all_points_with_global_scan_alignment.pkl` file.

## Setup

Run the following commands to install necessary dependencies.

```bash
  conda create -n ns3d python=3.7.11
  conda activate ns3d
  pip install -r requirements.txt
```

Install [Jacinle](https://github.com/vacancy/Jacinle).
```bash
  git clone https://github.com/vacancy/Jacinle --recursive
  export PATH=<path_to_jacinle>/bin:$PATH
```

Install the referit3d python package from [ReferIt3D](https://github.com/referit3d/referit3d).
```bash
  git clone https://github.com/referit3d/referit3d
  cd referit3d
  pip install -e .
```

Compile CUDA layers for [PointNet++](http://arxiv.org/abs/1706.02413).
```bash
  cd models/scene_graph/point_net_pp/pointnet2
  python setup.py install
```


## Evaluation

To evaluate our models and baseline models:

```bash

  scannet=<path_to/keep_all_points_with_global_scan_alignment.pkl>
  referit=<path_to/sr3d_train.csv>
  # baseline eva
  load_path=<path_to/baseline.pth>
  
  jac-run ns3d/trainval.py --desc ns3d/desc_ns3d.py --scannet-file $scannet --referit3D-file $referit --load $load_path --evaluate

  # Try1 eva
  load_path=<path_to/try1.pth>
  
  jac-run ns3d/trainval.py --desc ns3d/desc_symgraph.py --scannet-file $scannet --referit3D-file $referit --load $load_path --mode 'o' --evaluate

  # Try2 eva
  load_path=<path_to/try2.pth>
  
  jac-run ns3d/trainval.py --desc ns3d/desc_ns3d.py --scannet-file $scannet --referit3D-file $referit --load $load_path --mode 'ol' --evaluate
```

Weights for three models can be found at [trained.pth](https://drive.google.com/file/d/1pVgrXMpNRymhrp_xEZoGfVbXnW9Hcu7L/view?usp=sharing) and loaded into `load_path`.



## Training

To train baseline, NS3D:

```bash
  bash train.sh
```

To train Try1:

```bash
  bash train_ours.sh
```

To train Try2:

```bash
  bash train_ours_1.sh
```

Weights for our pretrained classification model can be found at [pretrained_cls.pth](https://drive.google.com/drive/folders/1NKFcxqb9OnfqZBgSSTLBiChntSl7svbs?usp=sharing) and loaded into `load_path`.



## Acknowledgements

Our codebase is built on top of [NSCL](https://github.com/vacancy/NSCL-PyTorch-Release), [NS3D](https://github.com/joyhsu0504/NS3D.git), and [ReferIt3D](https://github.com/referit3d/referit3d).
