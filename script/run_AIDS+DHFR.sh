CUDA_VISIBLE_DEVICES=4 python run.py -exp_type oodd -DS_pair AIDS+DHFR -batch_size_test 128 -num_epoch 400 -num_cluster 10 -alpha 0.2 -OE 1  -oe_loss_type baloss  -oe_weight -1 -gamma 2 -lam_range "[0.1,1.0]" -id_cluster_num 3 -ssl_epoch 50 -realoe_rate 0.5  -mixup_rate 0.5  -inter_rate 0.5



