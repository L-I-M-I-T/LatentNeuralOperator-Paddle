import scipy.io as scio
import numpy as np
import os
import argparse


parser = argparse.ArgumentParser(description="Latent Neural Operator PyTorch")
parser.add_argument("--data_name", type=str, required=True)
arg = parser.parse_args()


def load_Darcy(path, src_res, obj_res):
    matdata = scio.loadmat(path)
    y1 = matdata['coeff']
    y1 = y1[:, ::(src_res-1)//(obj_res-1), ::(src_res-1)//(obj_res-1)][:, :obj_res, :obj_res]
    y2 = matdata['sol']
    y2 =y2[:, ::(src_res-1)//(obj_res-1), ::(src_res-1)//(obj_res-1)][:, :obj_res, :obj_res]
    x = np.reshape(np.dstack(np.meshgrid(np.linspace(0, 1, obj_res), np.linspace(0, 1, obj_res))), (obj_res, obj_res, 2))
    x = np.expand_dims(x, axis=0)
    x = np.repeat(x, y1.shape[0], axis=0)
    y1 = np.expand_dims(y1, axis=3)
    y2 = np.expand_dims(y2, axis=3)
    return x, y1, y2


def load_NS2d(path, src_res):
    n_frame = 10
    matdata = scio.loadmat(path)        
    y1 = matdata['u'][:,:,:,:n_frame]
    y2 = matdata['u'][:,:,:,-n_frame:]
    x = np.reshape(np.dstack(np.meshgrid(np.linspace(0, 1, src_res), np.linspace(0, 1, src_res))), (src_res, src_res, 2))
    x = np.expand_dims(x, axis=0)
    x = np.repeat(x, y1.shape[0], axis=0)
    return x, y1, y2


def split_and_save(data_name, x, y1, y2, train_num, val_num):
    def split(data, train_num, val_num):
        train_data = data[:train_num]
        val_data = data[-val_num:]
        return train_data, val_data

    train_x, val_x = split(x, train_num, val_num)
    train_y1, val_y1 = split(y1, train_num, val_num)
    train_y2, val_y2 = split(y2, train_num, val_num)
    
    train_data = {"x":train_x, "y1":train_y1, "y2":train_y2}
    val_data = {"x":val_x, "y1":val_y1, "y2":val_y2}
    
    np.save("./datas/{}_train".format(data_name), train_data)
    np.save("./datas/{}_val".format(data_name), val_data)
    
    
if __name__ == "__main__":
    if not os.path.exists("./datas/{}_train.npy".format(arg.data_name)) or \
        not os.path.exists("./datas/{}_val.npy".format(arg.data_name)):
        print("Preparing data...")
        if arg.data_name == "Darcy":
            SRC_RES = 421
            OBJ_RES = 85
            TRAIN_NUM = 1000
            VAL_NUM = 200
            x, y1, y2 = load_Darcy("./datas/piececonst_r{}_N1024_smooth1.mat".format(SRC_RES), SRC_RES, OBJ_RES)
            tx, ty1, ty2 = load_Darcy("./datas/piececonst_r{}_N1024_smooth2.mat".format(SRC_RES), SRC_RES, OBJ_RES)
            x = np.concatenate((x, tx), axis=0)
            y1 = np.concatenate((y1, ty1), axis=0)
            y2 = np.concatenate((y2, ty2), axis=0)
            split_and_save(arg.data_name, x, y1, y2, TRAIN_NUM, VAL_NUM)
        elif arg.data_name == "NS2d":
            SRC_RES = 64
            TRAIN_NUM = 1000
            VAL_NUM = 200
            x, y1, y2 = load_NS2d("./datas/NavierStokes_V1e-5_N1200_T20", SRC_RES)
            split_and_save(arg.data_name, x, y1, y2, TRAIN_NUM, VAL_NUM)
        elif arg.data_name == "Airfoil":
            TRAIN_NUM = 1000
            VAL_NUM = 200
            Q = np.expand_dims(np.load("./datas/NACA_Cylinder_Q.npy")[:,4,:,:], axis=-1)
            X = np.expand_dims(np.load("./datas/NACA_Cylinder_X.npy"), axis=-1)
            Y = np.expand_dims(np.load("./datas/NACA_Cylinder_Y.npy"), axis=-1)
            x = np.concatenate((X, Y), axis=-1)
            y1 = x
            y2 = Q
            split_and_save(arg.data_name, x, y1, y2, TRAIN_NUM, VAL_NUM)
        elif arg.data_name == "Elasticity":
            TRAIN_NUM = 1000
            VAL_NUM = 200
            rr = np.load("./datas/Random_UnitCell_rr_10.npy")
            sigma = np.load("./datas/Random_UnitCell_sigma_10.npy")
            theta = np.load("./datas/Random_UnitCell_theta_10.npy")
            XY = np.load("./datas/Random_UnitCell_XY_10.npy")
            XY = np.transpose(XY, (2, 0, 1))
            sigma = np.transpose(sigma, (1, 0))
            sigma = np.expand_dims(sigma, axis=2)
            x = XY
            y1 = x
            y2 = sigma
            split_and_save(arg.data_name, x, y1, y2, TRAIN_NUM, VAL_NUM)
        elif arg.data_name == "Plasticity":
            SRC_RES1 = 101
            SRC_RES2 = 31
            SRC_RES3 = 20
            TRAIN_NUM = 900
            VAL_NUM = 87
            x = []
            for x1 in np.linspace(0, 1, SRC_RES1):
                for x2 in np.linspace(0, 1, SRC_RES2):
                    for x3 in np.linspace(0, 1, SRC_RES3):
                        x.append([x1, x2, x3])
            x = np.reshape(np.array(x), (SRC_RES1, SRC_RES2, SRC_RES3, 3))
            x = np.expand_dims(x, axis=0)
            x = np.repeat(x, TRAIN_NUM + VAL_NUM, axis=0)
            matdata = scio.loadmat("./datas/plas_N987_T20.mat")
            input = matdata["input"]
            output = matdata["output"]
            y1 = np.expand_dims(input, axis=-1)
            y1 = np.repeat(y1, SRC_RES2, axis=-1)
            y1 = np.expand_dims(y1, axis=-1)
            y1 = np.repeat(y1, SRC_RES3, axis=-1)
            y1 = np.expand_dims(y1, axis=-1)
            y2 = output
            split_and_save(arg.data_name, x, y1, y2, TRAIN_NUM, VAL_NUM)
        elif arg.data_name == "Pipe":
            TRAIN_NUM = 1000
            VAL_NUM = 200
            Q = np.expand_dims(np.load("./datas/Pipe_Q.npy")[:,0], axis=-1)
            X = np.expand_dims(np.load("./datas/Pipe_X.npy"), axis=-1)
            Y = np.expand_dims(np.load("./datas/Pipe_Y.npy"), axis=-1)
            x = np.concatenate((X, Y), axis=-1)
            y1 = x
            y2 = Q
            split_and_save(arg.data_name, x, y1, y2, TRAIN_NUM, VAL_NUM)
        else:
            raise NotImplementedError("Invalid data name !")
        print("Data preparing done.")
    else:
        print("Using prepared data.")
