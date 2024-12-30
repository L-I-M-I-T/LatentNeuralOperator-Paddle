config_name="LNO_time_NS2d"
exp_name="LNO_NS2d"
python prepare.py --data_name NS2d
torchrun \
--nnodes 1 \
--nproc_per_node 1 \
--master_port 12343 \
exp.py \
--config $config_name \
--device "3" \
--exp $exp_name \
--seed 0
