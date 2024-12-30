config_name="LNO_Plasticity"
exp_name="LNO_Plasticity"
python prepare.py --data_name Plasticity
torchrun \
--nnodes 1 \
--nproc_per_node 1 \
--master_port 12345 \
exp.py \
--config $config_name \
--device "5" \
--exp $exp_name \
--seed 0
