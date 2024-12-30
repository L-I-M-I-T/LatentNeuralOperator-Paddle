import argparse
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from module.utils import *

parser = argparse.ArgumentParser(description="Latent Neural Operator PyTorch")
parser.add_argument('--config', type=str, default=None, required=True)
parser.add_argument('--device', type=str, default=None, required=True)  
parser.add_argument('--seed',  type=int, default=0)
args = parser.parse_args()


def train_propagator(train_dataloader,
        val_dataloader,
        test_dataloader,
        transformer,
        masker,
        poser,
        model,
        model_name,
        loss_fn,
        optimizer,
        scheduler,
        local_rank,
        world_size,
        grad_clip,
        epoch,
        log_print_interval_epoch,
        model_save_interval_epoch,
        log_dir,
        checkpoint_dir):
    
    device = local_rank
    
    if local_rank == 0:
        print("Number of Propagator Parameters: {}".format(get_num_params(model)))
        print(model)
        print("Start Training...")
        writer = SummaryWriter(log_dir)
        checker = Checkpoint(checkpoint_dir, model, device)
        epoch_history = []
        lr_history = []
        train_loss_history = []
        val_loss_history = []
        test_metric_history = []
        test_side1_metric_history = []
        test_side2_metric_history = []
    
    for i in range(epoch):
        torch.distributed.barrier()
        train_dataloader.sampler.set_epoch(i)
        train_loss = 0

        for _, data in enumerate(tqdm(train_dataloader)):
            x, y, _ = data
            if "GNOT" in model_name:
                x, y, ob = data_preprocess_propagator_GNOT(masker, x, y, device)
            elif "LNO" in model_name:
                x, y, ob = data_preprocess_propagator_LNO(masker, x, y, device)
            elif "DeepONet" in model_name:
                x, y, ob = data_preprocess_propagator_DeepONet(masker, x, y, device)
            else:
                raise NotImplementedError("Invalid Propagator Name !")

            loss = torch.zeros((1)).to(device)
            optimizer.zero_grad()
            model.train()

            for j in range(0, len(ob)):
                if "time" in model_name:
                    t = torch.ones((x[j].shape[0], 1)) * j
                    t = t.to(device)
                    res = model(x[j], ob[j], t)
                else:
                    res = model(x[j], ob[j])
                
                loss = loss + loss_fn(res, y[j])

            loss = loss / len(ob)              
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            scheduler.step()
            
            train_batch_loss = torch.tensor(loss.item()).to(device)
            torch.distributed.all_reduce(train_batch_loss)
            train_loss = train_loss + train_batch_loss / world_size

        train_loss = train_loss / len(train_dataloader)
        train_loss = train_loss.item()
        val_loss = val_propagator(val_dataloader, transformer, masker, model, model_name, loss_fn, local_rank, world_size)
        val_loss = val_loss.item()
        test_metric, test_side1_metric, test_side2_metric = test_propagator(test_dataloader, transformer, masker, poser, model, model_name, local_rank, world_size)
        test_metric = test_metric.item()
        test_side1_metric = test_side1_metric.item()
        test_side2_metric = test_side2_metric.item()
        
        if local_rank == 0:
            if (i + 1) % log_print_interval_epoch == 0:
                writer.add_scalar("Learning Rate", optimizer.state_dict()['param_groups'][0]['lr'], i+1)
                writer.add_scalar("Train Loss", train_loss)
                writer.add_scalar("Val Loss", val_loss)
                writer.add_scalar("Test Metric", test_metric)
                writer.add_scalar("Test Side1 Metric", test_side1_metric)
                writer.add_scalar("Test Side2 Metric", test_side2_metric)
                
                epoch_history.append(i+1)
                lr_history.append(optimizer.state_dict()['param_groups'][0]['lr'])
                train_loss_history.append(train_loss)
                val_loss_history.append(val_loss)
                test_metric_history.append(test_metric)
                test_side1_metric_history.append(test_side1_metric)
                test_side2_metric_history.append(test_side2_metric)
                
                print("Epoch: {}\tLearning Rate :{}\tTrain Loss: {}\tVal Loss: {}\tTest Metric: {}\tTest Side1 Metric: {}\tTest Side2 Metric: {}"\
                    .format(i+1, optimizer.state_dict()['param_groups'][0]['lr'], train_loss, val_loss, test_metric, test_side1_metric, test_side2_metric))
            
            if (i + 1) % model_save_interval_epoch == 0:
                checker.save(i+1)
    
    if local_rank == 0:
        writer.close()
        logger(log_dir, 
               [epoch_history, lr_history, train_loss_history, val_loss_history, test_metric_history, test_side1_metric_history, test_side2_metric_history], 
               ["Epoch", "LR", "Train_Loss", "Val_Loss", "Test_Metric", "Test_Side1_Metric", "Test_Side2_Metric"])
        print("Finish Training !")


def train_completer(train_dataloader,
        val_dataloader,
        test_dataloader,
        transformer,
        masker,
        poser,
        model,
        model_name,
        loss_fn,
        optimizer,
        scheduler,
        local_rank,
        world_size,
        grad_clip,
        epoch,
        log_print_interval_epoch,
        model_save_interval_epoch,
        log_dir,
        checkpoint_dir):

    device = local_rank
    
    if local_rank == 0:
        print("Number of Completer Parameters: {}".format(get_num_params(model)))
        print(model)
        print("Start Training...")
        writer = SummaryWriter(log_dir)
        checker = Checkpoint(checkpoint_dir, model, device)
        epoch_history = []
        lr_history = []
        train_loss_history = []
        val_loss_history = []
        test_metric_history = []
        test_side1_metric_history = []
        test_side2_metric_history = []
    
    for i in range(epoch):
        torch.distributed.barrier()
        train_dataloader.sampler.set_epoch(i)
        train_loss = 0

        for _, data in enumerate(tqdm(train_dataloader)):
            x, y, _ = data
            if "DeepONet" in model_name:
                x, y, ob = data_preprocess_completer_DeepONet(masker, x, y, device)
            elif "GNOT" in model_name:
                x, y, ob = data_preprocess_completer_GNOT(masker, x, y, device)
            elif "LNO" in model_name:
                x, y, ob = data_preprocess_completer_LNO(masker, x, y, device)
            else:
                raise NotImplementedError("Invalid Completer Name")
           
            loss = torch.zeros((1)).to(device)
            optimizer.zero_grad()
            model.train()
            res = model(x, ob)
            loss = loss_fn(res, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            scheduler.step()
            
            train_batch_loss = torch.tensor(loss.item()).to(device)
            torch.distributed.all_reduce(train_batch_loss)
            train_loss = train_loss + train_batch_loss / world_size

        train_loss = train_loss / len(train_dataloader)
        train_loss = train_loss.item()
        val_loss = val_completer(val_dataloader, transformer, masker, model, model_name, loss_fn, local_rank, world_size)
        val_loss = val_loss.item()
        test_metric, test_side1_metric, test_side2_metric = test_completer(test_dataloader, transformer, masker, poser, model, model_name, local_rank, world_size)
        test_metric = test_metric.item()
        test_side1_metric = test_side1_metric.item()
        test_side2_metric = test_side2_metric.item()
        
        if local_rank == 0:
            if (i + 1) % log_print_interval_epoch == 0:
                writer.add_scalar("Learning Rate", optimizer.state_dict()['param_groups'][0]['lr'], i+1)
                writer.add_scalar("Train Loss", train_loss)
                writer.add_scalar("Val Loss", val_loss)
                writer.add_scalar("Test Metric", test_metric)
                writer.add_scalar("Test Side1 Metric", test_side1_metric)
                writer.add_scalar("Test Side2 Metric", test_side2_metric)
                
                epoch_history.append(i+1)
                lr_history.append(optimizer.state_dict()['param_groups'][0]['lr'])
                train_loss_history.append(train_loss)
                val_loss_history.append(val_loss)
                test_metric_history.append(test_metric)
                test_side1_metric_history.append(test_side1_metric)
                test_side2_metric_history.append(test_side2_metric)
                
                print("Epoch: {}\tLearning Rate :{}\tTrain Loss: {}\tVal Loss: {}\tTest Metric: {}\tTest Side1 Metric: {}\tTest Side2 Metric: {}"\
                    .format(i+1, optimizer.state_dict()['param_groups'][0]['lr'], train_loss, val_loss, test_metric, test_side1_metric, test_side2_metric))
            
            if (i + 1) % model_save_interval_epoch == 0:
                checker.save(i+1)
    
    if local_rank == 0:
        writer.close()
        logger(log_dir, 
               [epoch_history, lr_history, train_loss_history, val_loss_history, test_metric_history, test_side1_metric_history, test_side2_metric_history], 
               ["Epoch", "LR", "Train_Loss", "Val_Loss", "Test_Metric", "Test_Side1_Metric", "Test_Side2_Metric"])
        print("Finish Training !")


def val_propagator(val_dataloader,
        transformer,
        masker,
        model,
        model_name,
        loss_fn,
        local_rank,
        world_size):
    
    with torch.no_grad():
        device = local_rank
        val_loss = 0
        
        for _, data in enumerate(val_dataloader):
            x, y, _ = data
            if "GNOT" in model_name:
                x, y, ob = data_preprocess_propagator_GNOT(masker, x, y, device)
            elif "LNO" in model_name:
                x, y, ob = data_preprocess_propagator_LNO(masker, x, y, device)
            elif "DeepONet" in model_name:
                x, y, ob = data_preprocess_propagator_DeepONet(masker, x, y, device)
            else:
                raise NotImplementedError("Invalid Propagator Name !")
            
            loss = torch.zeros((1)).to(device)
            model.eval()
            for i in range(0, len(ob)):
                if "time" in model_name:
                    t = torch.ones((x[i].shape[0], 1)) * i
                    t = t.to(device)
                    res = model(x[i], ob[i], t)
                else:
                    res = model(x[i], ob[i])
                loss = loss + loss_fn(res, y[i])
            loss = loss / len(ob)
            val_batch_loss = torch.tensor(loss.item()).to(device)
            torch.distributed.all_reduce(val_batch_loss)
            val_loss = val_loss + val_batch_loss / world_size
        
        val_loss /= len(val_dataloader)

    return val_loss


def val_completer(val_dataloader,
        transformer,
        masker,
        model,
        model_name,
        loss_fn,
        local_rank,
        world_size):
    
    with torch.no_grad():
        device = local_rank
        val_loss = 0
        
        for _, data in enumerate(val_dataloader):
            x, y, _ = data
            if "DeepONet" in model_name:
                x, y, ob = data_preprocess_completer_DeepONet(masker, x, y, device)
            elif "GNOT" in model_name:
                x, y, ob = data_preprocess_completer_GNOT(masker, x, y, device)
            elif "LNO" in model_name:
                x, y, ob = data_preprocess_completer_LNO(masker, x, y, device)
            else:
                raise NotImplementedError("Invalid Completer Name")
            
            loss = torch.zeros((1)).to(device)
            model.eval()
            res = model(x, ob)
            loss = loss_fn(res, y)
            val_batch_loss = torch.tensor(loss.item()).to(device)
            torch.distributed.all_reduce(val_batch_loss)
            val_loss = val_loss + val_batch_loss / world_size
        
        val_loss /= len(val_dataloader)

    return val_loss


def test_propagator(test_dataloader,
        transformer,
        masker,
        poser,
        model,
        model_name,
        local_rank,
        world_size):

    with torch.no_grad():
        device = local_rank
        test_metric = 0
        test_side1_metric = 0
        test_side2_metric = 0
        
        for _, data in enumerate(test_dataloader):
            x, y, _ = data
            if "GNOT" in model_name:
                x, y, ob = data_preprocess_propagator_GNOT(masker, x, y, device)
            elif "LNO" in model_name:
                x, y, ob = data_preprocess_propagator_LNO(masker, x, y, device)
            elif "DeepONet" in model_name:
                x, y, ob = data_preprocess_propagator_DeepONet(masker, x, y, device)
            else:
                raise NotImplementedError("Invalid Propagator Name !")
            
            model.eval()
            if "time" in model_name:
                t = torch.zeros((x[0].shape[0], 1))
                t = t.to(device)
                res = model(x[0], ob[0], t)
            else:
                res = model(x[0], ob[0])
            
            for i in range(1, len(ob)):
                if "time" in model_name:
                    t = torch.ones((x[i].shape[0], 1)) * i
                    t = t.to(device)
                    res = model(x[i], ob[i], t)
                else:
                    res = model(x[i], ob[i])

            mask, _, _ = masker.get()
            res = torch.reshape(res, (res.shape[0], *tuple(mask.shape[1:]), res.shape[-1]))
            y = torch.reshape(y[-1], (y[-1].shape[0], *tuple(mask.shape[1:]), y[-1].shape[-1]))
            
            res = transformer.apply_y(res, inverse=True)
            y = transformer.apply_y(y, inverse=True)

            pos = poser.get()
            pos_idx = list(pos.to_sparse().indices().transpose(0,1).numpy())
            res = res.cpu().numpy()
            res = torch.tensor(np.array([res[(slice(None), *tuple(idx), slice(None))] for idx in pos_idx])).float().transpose(0, 1).to(device)
            y = y.cpu().numpy()
            y = torch.tensor(np.array([y[(slice(None), *tuple(idx), slice(None))] for idx in pos_idx])).float().transpose(0, 1).to(device)

            p = 2
            metric = RelLpLoss(p)
            test_batch_metric = torch.tensor(metric(res, y).item()).to(device)
            torch.distributed.all_reduce(test_batch_metric)
            test_metric = test_metric + test_batch_metric / world_size
            
            p = 2
            side1_metric = MpELoss(p)
            test_batch_side1_metric = torch.tensor(side1_metric(res, y).item()).to(device)
            torch.distributed.all_reduce(test_batch_side1_metric)
            test_side1_metric = test_side1_metric + test_batch_side1_metric / world_size
            
            p = 1  
            side2_metric = RelMpELoss(p)
            test_batch_side2_metric = torch.tensor(side2_metric(res, y).item()).to(device)
            torch.distributed.all_reduce(test_batch_side2_metric)
            test_side2_metric = test_side2_metric + test_batch_side2_metric / world_size
        
        test_metric /= len(test_dataloader)
        test_side1_metric /= len(test_dataloader)
        test_side2_metric /= len(test_dataloader)

    return test_metric, test_side1_metric, test_side2_metric


def test_completer(test_dataloader,
        transformer,
        masker,
        poser,
        model,
        model_name,
        local_rank,
        world_size):

    with torch.no_grad():
        device = local_rank
        test_metric = 0
        test_side1_metric = 0
        test_side2_metric = 0
        
        for _, data in enumerate(test_dataloader):
            x, y, _ = data
            if "DeepONet" in model_name:
                x, y, ob = data_preprocess_completer_DeepONet(masker, x, y, device)
            elif "GNOT" in model_name:
                x, y, ob = data_preprocess_completer_GNOT(masker, x, y, device)
            elif "LNO" in model_name:
                x, y, ob = data_preprocess_completer_LNO(masker, x, y, device)
            else:
                raise NotImplementedError("Invalid Completer Name")
            
            model.eval()
            res = model(x, ob)
            mask, vertex0, vertex1 = masker.get()
            res = torch.reshape(res, (res.shape[0], *tuple(vertex1 - vertex0), res.shape[-1]))
            y = torch.reshape(y, (y.shape[0], *tuple(vertex1 - vertex0), y.shape[-1]))
            
            res = transformer.apply_y(res, inverse=True)
            y = transformer.apply_y(y, inverse=True)

            pos = poser.get()
            pos_idx = list(pos.to_sparse().indices().transpose(0,1).numpy())
            res = res.cpu().numpy()
            res = torch.tensor(np.array([res[(slice(None), *tuple(idx), slice(None))] for idx in pos_idx])).float().transpose(0, 1).to(device)
            y = y.cpu().numpy()
            y = torch.tensor(np.array([y[(slice(None), *tuple(idx), slice(None))] for idx in pos_idx])).float().transpose(0, 1).to(device)

            p = 2
            metric = RelLpLoss(p)
            test_batch_metric = torch.tensor(metric(res, y).item()).to(device)
            torch.distributed.all_reduce(test_batch_metric)
            test_metric = test_metric + test_batch_metric / world_size
            
            p = 2
            side1_metric = MpELoss(p)
            test_batch_side1_metric = torch.tensor(side1_metric(res, y).item()).to(device)
            torch.distributed.all_reduce(test_batch_side1_metric)
            test_side1_metric = test_side1_metric + test_batch_side1_metric / world_size
            
            p = 1  
            side2_metric = RelMpELoss(p)
            test_batch_side2_metric = torch.tensor(side2_metric(res, y).item()).to(device)
            torch.distributed.all_reduce(test_batch_side2_metric)
            test_side2_metric = test_side2_metric + test_batch_side2_metric / world_size
        
        test_metric /= len(test_dataloader)
        test_side1_metric /= len(test_dataloader)
        test_side2_metric /= len(test_dataloader)

    return test_metric, test_side1_metric, test_side2_metric


if __name__ == "__main__":
    set_seed(args.seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device
    torch.distributed.init_process_group("nccl")
    local_rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    device = local_rank
    torch.cuda.set_device(device)

    config_file = "./configs/" + args.config + ".jsonc"
    config = Configuration(config_file)
    
    train_dataset, train_sampler, train_dataloader, \
    val_dataset, val_sampler, val_dataloader, \
    test_dataset, test_sampler, test_dataloader, \
    transformer, masker, poser, \
    model, loss, optimizer, scheduler \
    = get_data_model(config, device)
    
    log_dir = "./experiments/" + args.config + "/log/"
    checkpoint_dir = "./experiments/" + args.config + "/checkpoint/"

    if config.role == "propagator":
        train_propagator(
            train_dataloader,
            val_dataloader,
            test_dataloader,
            transformer,
            masker,
            poser,
            model,
            config.model.name,
            loss,
            optimizer,
            scheduler,
            local_rank,
            world_size,
            config.train.grad_clip,
            config.train.epoch,
            config.train.log_print_interval_epoch,
            config.train.model_save_interval_epoch,
            log_dir,
            checkpoint_dir
            )
    elif config.role == "completer":
        train_completer(
            train_dataloader,
            val_dataloader,
            test_dataloader,
            transformer,
            masker,
            poser,
            model,
            config.model.name,
            loss,
            optimizer,
            scheduler,
            local_rank,
            world_size,
            config.train.grad_clip,
            config.train.epoch,
            config.train.log_print_interval_epoch,
            config.train.model_save_interval_epoch,
            log_dir,
            checkpoint_dir
            )
    torch.distributed.destroy_process_group()
