import paddle
import argparse
from tqdm import tqdm
from module.loss import RelLpLoss
from module.model import LNO
from module.dataset import LNO_Dataset
from module.utils import get_num_params, save_para, set_seed


parser = argparse.ArgumentParser(description="Latent Neural Operator Paddle")

# data
parser.add_argument("--data_name", type=str, required=True)
parser.add_argument("--data_normalize", action="store_true")
parser.add_argument("--data_concat", action="store_true")
parser.add_argument("--train_batch_size", type=int, required=True)
parser.add_argument("--val_batch_size", type=int, required=True)

# model
parser.add_argument("--n_block", type=int, required=True)
parser.add_argument("--n_mode", type=int, required=True)
parser.add_argument("--n_dim", type=int, required=True)
parser.add_argument("--n_layer", type=int, required=True)
parser.add_argument("--n_head", type=int, required=True)
parser.add_argument("--trunk_dim", type=int, required=True)
parser.add_argument("--branch_dim", type=int, required=True)
parser.add_argument("--out_dim", type=int, required=True)

# optimization
parser.add_argument("--beta0", type=float, required=True)
parser.add_argument("--beta1", type=float, required=True)
parser.add_argument("--weight_decay", type=float, required=True)
parser.add_argument("--clip_norm", type=float, required=True)
parser.add_argument("--lr", type=float, required=True)
parser.add_argument("--div_factor", type=float, required=True)
parser.add_argument("--final_div_factor", type=float, required=True)
parser.add_argument("--pct_start", type=float, required=True)

# exp
parser.add_argument("--epoch", type=int, required=True)
parser.add_argument("--log_epoch", type=int, required=True)
parser.add_argument("--checkpoint_epoch", type=int, required=True)
parser.add_argument("--seed", type=int, required=True)
parser.add_argument("--exp_name", type=str, required=True)

arg = parser.parse_args()


if __name__ == "__main__":
    set_seed(arg.seed)
    
    train_dataset = LNO_Dataset(arg.data_name, "train", arg.data_normalize, arg.data_concat)
    train_dataloader = paddle.io.DataLoader(
        dataset=train_dataset , 
        batch_size=arg.train_batch_size, 
        drop_last=True, 
        shuffle=False)
    
    val_dataset = LNO_Dataset(arg.data_name, "val", arg.data_normalize, arg.data_concat)
    val_dataloader = paddle.io.DataLoader(
        dataset=val_dataset , 
        batch_size=arg.val_batch_size, 
        drop_last=True, 
        shuffle=False)
    
    loss_fn = RelLpLoss(p=2).to("gpu:0")
    
    model = LNO(n_block=arg.n_block, 
                n_mode=arg.n_mode, 
                n_dim=arg.n_dim, 
                n_head=arg.n_head, 
                n_layer=arg.n_layer, 
                trunk_dim=arg.trunk_dim, 
                branch_dim=arg.branch_dim, 
                out_dim=arg.out_dim).to("gpu:0")

    clip = paddle.nn.ClipGradByNorm(clip_norm=arg.clip_norm)
    
    scheduler = paddle.optimizer.lr.OneCycleLR(max_learning_rate=arg.lr, 
                                               divide_factor=arg.div_factor, 
                                               end_learning_rate=arg.lr / arg.div_factor / arg.final_div_factor, 
                                               phase_pct=arg.pct_start, 
                                               total_steps=len(train_dataloader)*arg.epoch)

    optimizer = paddle.optimizer.AdamW(learning_rate=scheduler, 
                                       parameters=model.parameters(), 
                                       beta1=arg.beta0, 
                                       beta2=arg.beta1, 
                                       grad_clip=clip, 
                                       weight_decay=arg.weight_decay)
    
    print("Start Training !")
    print("Total Parameter Number: {}".format(get_num_params(model)))
    print(model)
    save_para(arg, "./experiments/{}.json".format(arg.exp_name))

    for i in range(arg.epoch):
        train_loss = 0
        
        for data in tqdm(train_dataloader):
            x, y1, y2 = data

            x = x.to("gpu:0")
            x = paddle.reshape(x, (x.shape[0], -1, x.shape[-1]))
            y1 = y1.to("gpu:0")
            y1 = paddle.reshape(y1, (y1.shape[0], -1, y1.shape[-1]))
            y2 = y2.to("gpu:0")
            y2 = paddle.reshape(y2, (y2.shape[0], -1, y2.shape[-1]))

            model.train()
            
            pred = model(x, y1)
            loss = loss_fn(pred, y2)                
            optimizer.clear_gradients()
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            train_loss += loss.item()

        train_loss /= len(train_dataloader)

        with paddle.no_grad():
            val_loss = 0
            
            for data in tqdm(val_dataloader):
                x, y1, y2 = data
                
                x = x.to("gpu:0")
                x = paddle.reshape(x, (x.shape[0], -1, x.shape[-1]))
                y1 = y1.to("gpu:0")
                y1 = paddle.reshape(y1, (y1.shape[0], -1, y1.shape[-1]))
                y2 = y2.to("gpu:0")
                y2 = paddle.reshape(y2, (y2.shape[0], -1, y2.shape[-1]))
                
                model.eval()
                
                pred = model(x, y1)
                if arg.data_normalize:
                    pred = train_dataset.normalizer.apply_y2(pred, "gpu:0", inverse=True)
                    y2 = train_dataset.normalizer.apply_y2(y2, "gpu:0", inverse=True)
                
                loss = loss_fn(pred, y2)               
                val_loss += loss.item()
            
            val_loss /= len(val_dataloader)
        
        if (i+1) % arg.log_epoch == 0:
            print("Epoch: {}\tLR: {}\tTrain Loss: {}\tVal Loss: {}".format(i+1, optimizer._global_learning_rate().numpy(), train_loss, val_loss))
            
        if (i+1) % arg.checkpoint_epoch == 0:
            paddle.save(model.state_dict(), "./experiments/{}/{}.pt".format(arg.exp_name, i+1))

    print("Training Done !")
