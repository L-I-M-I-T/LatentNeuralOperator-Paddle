config_name="LNO_propagator_Burgers"
python prepare.py --data_name Burgers_IC_std
torchrun \
--nnodes 1 \
--nproc_per_node 2 \
--master_port 12347 \
train.py \
--config $config_name \
--device "6,7" \
--seed 0
