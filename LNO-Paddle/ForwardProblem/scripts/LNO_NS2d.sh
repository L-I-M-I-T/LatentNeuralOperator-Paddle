python prepare.py --data_name NS2d
CUDA_VISIBLE_DEVICES="3" \
python exp_time.py \
--T 10 \
--step 1 \
--data_name "NS2d" \
--data_concat \
--train_batch_size 4 \
--val_batch_size 4 \
--n_block 8 \
--n_mode 256 \
--n_dim 256 \
--n_layer 3 \
--n_head 8 \
--trunk_dim 2 \
--branch_dim 12 \
--out_dim 1 \
--beta0 0.9 \
--beta1 0.99 \
--weight_decay 0.00005 \
--clip_norm 1000.0 \
--lr 0.001 \
--div_factor 10000.0 \
--final_div_factor 10000.0 \
--pct_start 0.2 \
--epoch 500 \
--log_epoch 1 \
--checkpoint_epoch 50 \
--seed 0 \
--exp_name "LNO_NS2d"
