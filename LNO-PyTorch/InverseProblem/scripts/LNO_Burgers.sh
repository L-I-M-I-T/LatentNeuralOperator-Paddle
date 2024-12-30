completer_config_name="LNO_completer_fix_Burgers"
completer_epoch_num=500
propagator_config_name="LNO_propagator_Burgers"
propagator_epoch_num=500

torchrun \
--nnodes 1 \
--nproc_per_node 1 \
--master_port 12345 \
infer.py \
--config_completer $completer_config_name \
--epoch_completer $completer_epoch_num \
--config_propagator $propagator_config_name \
--epoch_propagator $propagator_epoch_num \
--device "0" \
--seed 0
