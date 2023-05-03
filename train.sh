export CUDA_VISIBLE_DEVICES=4
scannet=/public/MARS/datasets/ScanNet/keep_all_points_with_global_scan_alignment.pkl
referit=datasets/referit3d/data/scannet/sr3d_train.csv
load_path=pretrained_cls.pth

jac-run ns3d/trainval.py --desc ns3d/desc_ns3d.py --scannet-file $scannet --referit3D-file $referit --load $load_path --lr 0.0001 --epochs 5000 --save-interval 1 --validation-interval 1