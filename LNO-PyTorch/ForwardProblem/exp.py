import argparse
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from module.dataset import *
from module.model import *
from module.utils import *
from module.loss import *
import shutil


parser = argparse.ArgumentParser(description="Latent Neural Operator PyTorch")
parser.add_argument("--config", type=str, default=None, required=True)
parser.add_argument("--device", type=str, default=None, required=True)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--exp", type=str, default=None, required=True)
arg = parser.parse_args()


def train(train_dataloader,
          val_dataloader,
          transformer,
          model,
          model_attr,
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
        writer = SummaryWriter(log_dir)
        logger = Logger(log_dir)
        checker = Checkpoint(checkpoint_dir, model, device)
        epoch_history = []
        lr_history = []
        train_loss_history = []
        val_loss_history = []
        print("Number of Model Parameters: {}".format(get_num_params(model)))
        print(model)
        print("Start Training...")
        logger.print("Number of Model Parameters: {}".format(get_num_params(model)))
        logger.print(model)
        logger.print("Start Training...")
    
    for i in range(epoch):
        torch.distributed.barrier()
        train_dataloader.sampler.set_epoch(i)
        train_loss = 0

        for data in tqdm(train_dataloader):
            x, y1, y2 = data

            x = x.to(device)
            x = torch.reshape(x, (x.shape[0], -1, x.shape[-1]))
            y1 = y1.to(device)
            y1 = torch.reshape(y1, (y1.shape[0], -1, y1.shape[-1]))
            y2 = y2.to(device)
            y2 = torch.reshape(y2, (y2.shape[0], -1, y2.shape[-1]))

            model.train()
            if model_attr["single"]:
                res = model(y1)
            else:
                res = model(x, y1)
            
            loss = loss_fn(res, y2)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            scheduler.step()
            
            train_batch_loss = torch.tensor(loss.item()).to(device)
            torch.distributed.all_reduce(train_batch_loss)
            train_loss = train_loss + train_batch_loss / world_size
        

        train_loss = train_loss / len(train_dataloader)
        train_loss = train_loss.item()
        val_loss = val(val_dataloader, transformer, model, model_attr, loss_fn, local_rank, world_size)
        val_loss = val_loss.item()
        
        if local_rank == 0:
            if (i + 1) % log_print_interval_epoch == 0:
                writer.add_scalar("Learning Rate", optimizer.state_dict()['param_groups'][0]['lr'], i+1)
                writer.add_scalar("Train Loss", train_loss)
                writer.add_scalar("Val Loss", val_loss)
                
                epoch_history.append(i+1)
                lr_history.append(optimizer.state_dict()['param_groups'][0]['lr'])
                train_loss_history.append(train_loss)
                val_loss_history.append(val_loss)
                
                print("Epoch: {}\tLearning Rate :{}\tTrain Loss: {}\tVal Loss: {}".format(i+1, optimizer.state_dict()['param_groups'][0]['lr'], train_loss, val_loss))
                logger.print("Epoch: {}\tLearning Rate :{}\tTrain Loss: {}\tVal Loss: {}".format(i+1, optimizer.state_dict()['param_groups'][0]['lr'], train_loss, val_loss))
            
            if (i + 1) % model_save_interval_epoch == 0:
                checker.save(i+1)
    
    if local_rank == 0:
        writer.close()
        logger.save([epoch_history, lr_history, train_loss_history, val_loss_history], ["Epoch", "LR", "Train_Loss", "Val_Loss"])
        print("Finish Training !")
        logger.print("Finish Training !")


def val(val_dataloader,
        transformer,
        model,
        model_attr,
        loss_fn,
        local_rank,
        world_size):
    
    with torch.no_grad():
        device = local_rank
        val_loss = 0
        
        for data in tqdm(val_dataloader):
            x, y1, y2 = data
            
            x = x.to(device)
            x = torch.reshape(x, (x.shape[0], -1, x.shape[-1]))
            y1 = y1.to(device)
            y1 = torch.reshape(y1, (y1.shape[0], -1, y1.shape[-1]))
            y2 = y2.to(device)
            y2 = torch.reshape(y2, (y2.shape[0], -1, y2.shape[-1]))
            
            model.eval()
            if model_attr["single"]:
                res = model(y1)
            else:
                res = model(x, y1)

            if transformer.is_apply_y2():
                res = transformer.apply_y2(res, inverse=True)
                y2 = transformer.apply_y2(y2, inverse=True)
            
            loss = loss_fn(res, y2)
            
            val_batch_loss = torch.tensor(loss.item()).to(device)
            torch.distributed.all_reduce(val_batch_loss)
            val_loss = val_loss + val_batch_loss / world_size
        
        val_loss /= len(val_dataloader)

    return val_loss


def train_time(train_dataloader,
          val_dataloader,
          transformer,
          model,
          model_attr,
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
        writer = SummaryWriter(log_dir)
        logger = Logger(log_dir)
        checker = Checkpoint(checkpoint_dir, model, device)
        epoch_history = []
        lr_history = []
        train_loss_step_history = []
        train_loss_full_history = []
        val_loss_step_history = []
        val_loss_full_history = []
        print("Number of Model Parameters: {}".format(get_num_params(model)))
        print(model)
        print("Start Training...")
        logger.print("Number of Model Parameters: {}".format(get_num_params(model)))
        logger.print(model)
        logger.print("Start Training...")
    
    for i in range(epoch):
        torch.distributed.barrier()
        train_dataloader.sampler.set_epoch(i)
        train_loss_step = 0
        train_loss_full = 0

        for data in tqdm(train_dataloader):
            x, y1, y2 = data
            
            x = x.to(device)
            x = torch.reshape(x, (x.shape[0], -1, x.shape[-1]))
            y1 = y1.to(device)
            y1 = torch.reshape(y1, (y1.shape[0], -1, y1.shape[-1]))
            y2 = y2.to(device)
            y2 = torch.reshape(y2, (y2.shape[0], -1, y2.shape[-1]))

            model.train()
            
            T = 10
            step = 1
            loss = 0
            for t in range(0, T, step):
                gt = y2[..., t:t+step]
                if model_attr["single"]:
                    pred_step = model(y1)
                else:
                    pred_step = model(x, y1)
                    
                loss += loss_fn(pred_step, gt)
                
                if t == 0:
                    pred_full = pred_step
                else:
                    pred_full = torch.cat((pred_full, pred_step), -1)
                y1 = torch.cat((y1[..., :2], y1[..., 2+step:], gt), dim = -1)
            
            train_loss_step_temp = loss
            train_loss_full_temp = loss_fn(pred_full, y2)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            scheduler.step()
            
            train_batch_loss_step = torch.tensor(train_loss_step_temp.item()).to(device)
            torch.distributed.all_reduce(train_batch_loss_step)
            train_loss_step = train_loss_step + train_batch_loss_step / world_size
            
            train_batch_loss_full = torch.tensor(train_loss_full_temp.item()).to(device)
            torch.distributed.all_reduce(train_batch_loss_full)
            train_loss_full = train_loss_full + train_batch_loss_full / world_size
        
        
        train_loss_step = (train_loss_step / len(train_dataloader)).item()
        train_loss_full = (train_loss_full / len(train_dataloader)).item()
        
        val_loss_step, val_loss_full = val_time(val_dataloader, transformer, model, model_attr, loss_fn, local_rank, world_size)
        val_loss_step = val_loss_step.item()
        val_loss_full = val_loss_full.item()
        
        if local_rank == 0:
            if (i + 1) % log_print_interval_epoch == 0:
                writer.add_scalar("Learning Rate", optimizer.state_dict()['param_groups'][0]['lr'], i+1)
                writer.add_scalar("Train Loss Step", train_loss_step)
                writer.add_scalar("Train Loss Full", train_loss_full)
                
                writer.add_scalar("Val Loss Step", val_loss_step)
                writer.add_scalar("Val Loss Full", val_loss_full)
                
                epoch_history.append(i+1)
                lr_history.append(optimizer.state_dict()['param_groups'][0]['lr'])
                train_loss_step_history.append(train_loss_step)
                train_loss_full_history.append(train_loss_full)
                val_loss_step_history.append(val_loss_step)
                val_loss_full_history.append(val_loss_full)
                
                print("Epoch: {}\tLearning Rate :{}\tTrain Loss Step: {}\tTrain Loss Full: {}\tVal Loss Step: {}\tVal Loss Full: {}".format(i+1, optimizer.state_dict()['param_groups'][0]['lr'], train_loss_step, train_loss_full, val_loss_step, val_loss_full))
                logger.print("Epoch: {}\tLearning Rate :{}\tTrain Loss Step: {}\tTrain Loss Full: {}\tVal Loss Step: {}\tVal Loss Full: {}".format(i+1, optimizer.state_dict()['param_groups'][0]['lr'], train_loss_step, train_loss_full, val_loss_step, val_loss_full))
            
            if (i + 1) % model_save_interval_epoch == 0:
                checker.save(i+1)
    
    if local_rank == 0:
        writer.close()
        logger.save([epoch_history, lr_history, train_loss_step_history, train_loss_full_history, val_loss_step_history, val_loss_full_history], ["Epoch", "LR", "Train_Loss_Step", "Train_Loss_Full", "Val_Loss_Step", "Val_Loss_Full"])
        print("Finish Training !")
        logger.print("Finish Training !")


def val_time(val_dataloader,
        transformer,
        model,
        model_attr,
        loss_fn,
        local_rank,
        world_size):
    
    with torch.no_grad():
        device = local_rank
        val_loss_step = 0
        val_loss_full = 0
        
        for data in tqdm(val_dataloader):
            x, y1, y2 = data
            
            x = x.to(device)
            x = torch.reshape(x, (x.shape[0], -1, x.shape[-1]))
            y1 = y1.to(device)
            y1 = torch.reshape(y1, (y1.shape[0], -1, y1.shape[-1]))
            y2 = y2.to(device)
            y2 = torch.reshape(y2, (y2.shape[0], -1, y2.shape[-1]))
            
            model.eval()
            
            T = 10
            step = 1
            loss = 0
            for t in range(0, T, step):
                gt = y2[..., t:t+step]
                if model_attr["single"]:
                    pred_step = model(y1)
                else:
                    pred_step = model(x, y1)
                    
                loss += loss_fn(pred_step, gt)
                
                if t == 0:
                    pred_full = pred_step
                else:
                    pred_full = torch.cat((pred_full, pred_step), -1)
                y1 = torch.cat((y1[..., :2], y1[..., 2+step:], pred_step), dim=-1)
            
            val_loss_step_temp = loss
            val_loss_full_temp = loss_fn(pred_full, y2)
            
            val_batch_loss_step = torch.tensor(val_loss_step_temp.item()).to(device)
            torch.distributed.all_reduce(val_batch_loss_step)
            val_loss_step = val_loss_step + val_batch_loss_step / world_size
            
            val_batch_loss_full = torch.tensor(val_loss_full_temp.item()).to(device)
            torch.distributed.all_reduce(val_batch_loss_full)
            val_loss_full = val_loss_full + val_batch_loss_full / world_size
        
        val_loss_step /= len(val_dataloader)
        val_loss_full /= len(val_dataloader)

    return val_loss_step, val_loss_full


if __name__ == "__main__":
    set_seed(arg.seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = arg.device
    torch.distributed.init_process_group("nccl")
    local_rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    device = local_rank
    torch.cuda.set_device(device)

    config_file = "./configs/{}.jsonc".format(arg.config)
    config = Configuration(config_file)
    
    model_attr = dict()
    if "_time" in arg.config:
        model_attr["time"] = True
    else:
        model_attr["time"] = False
    
    train_dataloader, val_dataloader, _, \
    transformer, model, loss, optimizer, scheduler \
    = get_model_data(config, model_attr, device)
    
    log_dir = "./experiments/{}/log/".format(arg.exp)
    checkpoint_dir = "./experiments/{}/checkpoint/".format(arg.exp)
    src_dir = "./experiments/{}/src/".format(arg.exp)
    save_para(arg, config)
    
    if not os.path.exists(src_dir):
        os.makedirs(src_dir)
    for obj in os.listdir("."):
        if os.path.isfile(obj):
            shutil.copy(obj, src_dir + obj)
    
    if "_single" in config.model.name:
        model_attr["single"] = True
    else:
        model_attr["single"] = False
        
    if model_attr["time"]:
        train_time(
            train_dataloader,
            val_dataloader,
            transformer,
            model,
            model_attr,
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
    else:
        train(
            train_dataloader,
            val_dataloader,
            transformer,
            model,
            model_attr,
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
