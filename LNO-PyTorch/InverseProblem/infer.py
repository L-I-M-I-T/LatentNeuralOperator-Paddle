import argparse
from module.utils import *


parser = argparse.ArgumentParser(description="Latent Neural Operator PyTorch")
parser.add_argument('--config_completer', type=str, default=None, required=True)
parser.add_argument('--epoch_completer', type=int, default=None, required=True)
parser.add_argument('--config_propagator', type=str, default=None, required=True)
parser.add_argument('--epoch_propagator', type=int, default=None, required=True)
parser.add_argument('--device', type=str, default=None, required=True)
parser.add_argument('--seed', type=int, default=2023)
args = parser.parse_args()


def infer(infer_dataloader,
        transformer,
        masker_completer,
        masker_propagator,
        poser,
        completer,
        completer_name,
        propagator,
        propagator_name,
        local_rank,
        world_size):

    with torch.no_grad():
        device = local_rank
        infer_metric = 0
        infer_side1_metric = 0
        infer_side2_metric = 0
        
        for no, data in enumerate(infer_dataloader):
            source_x, source_y, _ = data
            if "DeepONet" in completer_name:
                x, y, ob = data_preprocess_completer_DeepONet(masker_completer, source_x.clone(), source_y.clone(), device)
            elif "GNOT" in completer_name:
                x, y, ob = data_preprocess_completer_GNOT(masker_completer, source_x.clone(), source_y.clone(), device)
            elif "LNO" in completer_name:
                x, y, ob = data_preprocess_completer_LNO(masker_completer, source_x.clone(), source_y.clone(), device)
            else:
                raise NotImplementedError("Invalid Completer Name !")
            
            completer.eval()
            res = completer(x, ob)
            res = torch.reshape(res, (res.shape[0], math.prod(tuple(res.shape[1:-1])), res.shape[-1]))
            x = torch.reshape(x, (x.shape[0], math.prod(tuple(x.shape[1:-1])), x.shape[-1]))
            y = torch.reshape(y, (y.shape[0], math.prod(tuple(y.shape[1:-1])), y.shape[-1]))
            
            res = torch.cat((x, res), dim=-1)
            
            if "GNOT" in propagator_name:
                x, y, ob = data_preprocess_propagator_GNOT(masker_propagator, source_x, source_y, device)
            elif "LNO" in propagator_name:
                x, y, ob = data_preprocess_propagator_LNO(masker_propagator, source_x, source_y, device)
            elif "DeepONet" in propagator_name:
                x, y, ob = data_preprocess_propagator_DeepONet(masker_propagator, source_x, source_y, device)
            else:
                raise NotImplementedError("Invalid Propagator Name !")

            propagator.eval()
            if "time" in propagator_name:
                t = torch.zeros((x[0].shape[0], 1))
                t = t.to(device)
                res = propagator(x[0], res, t)
            else:
                res = propagator(x[0], res)

            for i in range(1, len(ob)):
                ob[i] = torch.cat((x[i-1], res), dim=-1)
                if "time" in propagator_name:
                    t = torch.ones((x[i].shape[0], 1)) * i
                    t = t.to(device)
                    res = propagator(x[i], ob[i], t)
                else:
                    res = propagator(x[i], ob[i])

            mask, _, _ = masker_propagator.get()
            res = torch.reshape(res, (res.shape[0], *tuple(mask.shape[1:]), res.shape[-1]))
            y = torch.reshape(y[-1], (y[-1].shape[0], *tuple(mask.shape[1:]), y[-1].shape[-1]))
            
            res = transformer.apply_y(res, inverse=True)
            y = transformer.apply_y(y, inverse=True)
            
            np.save("./pics/res_npy/res{}".format(no), res[0].cpu().numpy())
            
            pos = poser.get()
            pos_idx = list(pos.to_sparse().indices().transpose(0,1).numpy())
            res = res.cpu().numpy()
            res = torch.tensor(np.array([res[(slice(None), *tuple(idx), slice(None))] for idx in pos_idx])).float().transpose(0, 1).to(device)
            y = y.cpu().numpy()
            y = torch.tensor(np.array([y[(slice(None), *tuple(idx), slice(None))] for idx in pos_idx])).float().transpose(0, 1).to(device)

            p = 2
            metric = RelLpLoss(p)
            infer_batch_metric = torch.tensor(metric(res, y).item()).to(device)
            torch.distributed.all_reduce(infer_batch_metric)
            infer_metric = infer_metric + infer_batch_metric / world_size
            
            p = 2
            side1_metric = MpELoss(p)
            infer_batch_side1_metric = torch.tensor(side1_metric(res, y).item()).to(device)
            torch.distributed.all_reduce(infer_batch_side1_metric)
            infer_side1_metric = infer_side1_metric + infer_batch_side1_metric / world_size
            
            p = 1  
            side2_metric = RelMpELoss(p)
            infer_batch_side2_metric = torch.tensor(side2_metric(res, y).item()).to(device)
            torch.distributed.all_reduce(infer_batch_side2_metric)
            infer_side2_metric = infer_side2_metric + infer_batch_side2_metric / world_size
        
        infer_metric /= len(infer_dataloader)
        infer_side1_metric /= len(infer_dataloader)
        infer_side2_metric /= len(infer_dataloader)

    return infer_metric, infer_side1_metric, infer_side2_metric


if __name__ == "__main__":    
    set_seed(args.seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device
    torch.distributed.init_process_group("nccl")
    local_rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    device = local_rank
    torch.cuda.set_device(device)

    config_completer_file = "./configs/" + args.config_completer + ".jsonc"
    config_completer = Configuration(config_completer_file)

    config_propagator_file = "./configs/" + args.config_propagator + ".jsonc"
    config_propagator = Configuration(config_propagator_file)
    
    _, _, train_dataloader, \
    _, _, val_dataloader, \
    _, _, test_dataloader, \
    transformer, masker_completer, _, \
    completer, _, _, _ \
    = get_data_model(config_completer, device)
    
    _, _, _, \
    _, _, _, \
    _, _, _, \
    _, masker_propagator, poser, \
    propagator, _, _, _ \
    = get_data_model(config_propagator, device)
    
    checkpoint_completer_dir = "./experiments/" + args.config_completer + "/checkpoint/"
    completer.load_state_dict(torch.load(checkpoint_completer_dir + str(args.epoch_completer) + ".pt", map_location="cuda:{}".format(device)))

    checkpoint_propagator_dir = "./experiments/" + args.config_propagator + "/checkpoint/"
    propagator.load_state_dict(torch.load(checkpoint_propagator_dir + str(args.epoch_propagator) + ".pt", map_location="cuda:{}".format(device)))

    infer_metric, infer_side1_metric, infer_side2_metric = \
    infer(test_dataloader, transformer, masker_completer, masker_propagator, poser, 
          completer, config_completer.model.name, propagator, config_propagator.model.name, local_rank, world_size)
    
    infer_metric = infer_metric.item()
    infer_side1_metric = infer_side1_metric.item()
    infer_side2_metric = infer_side2_metric.item()
    
    print("Test Metric: {}\tTest Side1 Metric: {}\tTest Side2 Metric: {}"\
        .format(infer_metric, infer_side1_metric, infer_side2_metric))
    torch.distributed.destroy_process_group()
