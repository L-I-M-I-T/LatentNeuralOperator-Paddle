config_name="LNO_Elasticity"
exp_name="LNO_Elasticity"
python prepare.py --data_name Elasticity
torchrun \
--nnodes 1 \
--nproc_per_node 1 \
--master_port 12342 \
exp.py \
--config $config_name \
--device "2" \
--exp $exp_name \
--seed 0
