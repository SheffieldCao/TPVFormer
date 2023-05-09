# vis frames by my model
python visualization/vis_frame.py --py-config config/tpv04_occupancy-vox_4e-1-dim_128-base.py \
                                --work-dir work_dirs/tpv04_occupancy-vox_4e-1-dim_128-base \
                                --ckpt-path work_dirs/tpv04_occupancy-vox_4e-1-dim_128-base/epoch_16.pth \
                                --save-path work_dirs/tpv04_occupancy-vox_4e-1-dim_128-base/vis \
                                --frame-idx 0 1 2 3 4 5 6