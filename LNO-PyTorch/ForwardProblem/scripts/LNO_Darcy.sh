config_name="LNO_Darcy"
exp_name="LNO_Darcy"
python prepare.py --data_name Darcy
torchrun \
--nnodes 1 \
--nproc_per_node 1 \
--master_port 12341 \
exp.py \
--config $config_name \
--device "1" \
--exp $exp_name \
--seed 0
