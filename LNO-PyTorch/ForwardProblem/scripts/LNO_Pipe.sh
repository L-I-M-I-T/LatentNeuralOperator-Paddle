config_name="LNO_Pipe"
exp_name="LNO_Pipe"
python prepare.py --data_name Pipe
torchrun \
--nnodes 1 \
--nproc_per_node 1 \
--master_port 12344 \
exp.py \
--config $config_name \
--device "4" \
--exp $exp_name \
--seed 0
