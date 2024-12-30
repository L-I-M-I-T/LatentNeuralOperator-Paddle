config_name="LNO_Airfoil"
exp_name="LNO_Airfoil"
python prepare.py --data_name Airfoil
torchrun \
--nnodes 1 \
--nproc_per_node 1 \
--master_port 12340 \
exp.py \
--config $config_name \
--device "0" \
--exp $exp_name \
--seed 0
