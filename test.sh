export CUDA_VISIBLE_DEVICES=0
scannet=/public/MARS/datasets/ScanNet/keep_all_points_with_global_scan_alignment.pkl
referit=datasets/referit3d/data/scannet/sr3d_train.csv
load_path=dumps/ns3d/desc_ns3d/default/reproduce/checkpoints/epoch_16.pth

jac-debug ns3d/trainval.py --desc ns3d/desc_ns3d.py --scannet-file $scannet --referit3D-file $referit --load $load_path --evaluate
