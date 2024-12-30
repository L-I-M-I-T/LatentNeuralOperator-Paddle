from scipy import interpolate
from sklearn import gaussian_process as gp
from scipy.integrate import odeint
import numpy as np
import os
import argparse


class GRF(object):
    def __init__(self, left, right, kernel="PER", length_scale=1, N=1000, interp="cubic", processes=4):
        self.N = N
        self.interp = interp
        self.x = np.linspace(left, right, num=N)[:, None]
        self.xx = np.linspace(left, right, num=N+1)[:, None]
        if kernel == "PER":
            K = gp.kernels.ExpSineSquared(length_scale=length_scale, periodicity=right-left)
        elif kernel == "AE":
            K = gp.kernels.Matern(length_scale=length_scale, nu=0.5)
        elif kernel == "RBF":
            K = gp.kernels.RBF(length_scale=length_scale)
            
        self.K = K(self.x)
        self.L = np.linalg.cholesky(self.K + 1e-13 * np.eye(self.N))
        self.processes = processes

    def feature(self, features_num, mean=0):
        u = np.random.randn(self.N, features_num)
        return (np.dot(self.L, u) + mean).T

    def eval(self, ys, sensors):
        if self.interp == "cubic":
            res = map(lambda y: interpolate.CubicSpline(np.ravel(self.xx), y, bc_type="periodic")(sensors).T, np.concatenate((ys, ys[:,0].reshape(-1,1)), axis=1))
        return np.vstack(list(res))[:,0:-1]


class PDE():
    def __init__(self, data_name, NX, NT, X0, X1, T):
        self.data_name = data_name
        self.NX = NX
        self.NT = NT
        self.X0 = X0
        self.X1 = X1
        self.T = T
        self.dx = (self.X1 - self.X0) / (self.NX - 1)
        self.dt = self.T / (self.NT - 1)
        self.x = np.linspace(self.X0, self.X1, self.NX)
        self.xx = np.linspace(self.X0, self.X1, self.NX + 1)
        self.t = np.linspace(0, self.T, self.NT)
        self.D = 0.01
        
        self.lap = np.diag(-2*np.ones(self.NX), k=0) + np.diag(np.ones(self.NX-1), k=-1) + np.diag(np.ones(self.NX-1), k=1)
        self.lap[0,-1] = 1
        self.lap[-1,0] = 1
        self.lap /= self.dx ** 2
        
        self.space_x = GRF(left=self.X0, right=self.X1)
    
    def ode_AllenCahn(self, u, t, F):
        return self.D * np.matmul(self.lap, u) - 0.01 * u * u + F

    def ode_Burgers(self, u, t, F):
        a_plus = np.maximum(u, 0)
        a_min = np.minimum(u, 0)
        u_left = np.concatenate((np.array([u[-1]]), u[:-1]))
        u_right = np.concatenate((u[1:], np.array([u[0]])))
        return self.D * np.matmul(self.lap, u) - (a_plus * (u - u_left) / self.dx + a_min * (u_right - u) / self.dx) + F
    
    def sample(self, f_num):
        f = []
        ob = []
        
        if "Force" in self.data_name:
            self.F = self.space_x.eval(self.space_x.feature(f_num), self.xx[:,None])
        else:
            self.F = np.zeros((f_num, self.NX))
        
        self.IC = self.space_x.eval(self.space_x.feature(f_num), self.xx[:,None])
        
        x = np.dstack(np.meshgrid(self.x, self.t)).reshape((self.NT, self.NX, 2))
        
        for i in range(f_num):
            if "Burgers" in self.data_name:
                y = odeint(self.ode_Burgers, self.IC[i], self.t, args=(self.F[i],))
            elif "AllenCahn" in self.data_name:
                y = odeint(self.ode_AllenCahn, self.IC[i], self.t, args=(self.F[i],))
            else:
                raise NotImplementedError("Invalid PDE type !")
        
            ob.append(y)
            f.append(self.F[i])

        ob = np.array(ob)
        f = np.array(f)
        return x, ob, f


def data_generation(data_name):
    if "_fine" in data_name:
        NX=512
        NT=512
        X0 = 0
        X1 = 1
        T = 1
    else:
        NX=512
        NT=512
        X0 = 0
        X1 = 1
        T = 1

    if "_big" in data_name:
        train_num = 16384
        val_num = 512
        test_num = 512
    elif "_std" in data_name:
        train_num = 4096
        val_num = 128
        test_num = 128
    elif "_small" in data_name:
        train_num = 1024
        val_num = 32
        test_num = 32
    elif "_try" in data_name:
        train_num = 16
        val_num = 4
        test_num = 4

    system = PDE(data_name, NX, NT, X0, X1, T)
    x, y, f = system.sample(train_num + val_num + test_num)
    
    x = np.expand_dims(x, axis=0)
    x = np.repeat(x, train_num + val_num + test_num, axis=0)
    y = np.expand_dims(y, axis=3)
    
    f = np.expand_dims(f, axis=1)
    f = np.repeat(f, NT, axis=1)
    f = np.expand_dims(f, axis=3)

    data = {"x": x[0:train_num], "y": y[0:train_num],"f": f[0:train_num]}
    np.save("./datas/{}_train.npy".format(data_name), data)

    data = {"x": x[train_num:train_num+val_num], "y": y[train_num:train_num+val_num],"f": f[train_num:train_num+val_num]}
    np.save("./datas/{}_val.npy".format(data_name), data)

    data = {"x": x[train_num+val_num:train_num+val_num+test_num], "y": y[train_num+val_num:train_num+val_num+test_num],"f": f[train_num+val_num:train_num+val_num+test_num]}
    np.save("./datas/{}_test.npy".format(data_name), data)


parser = argparse.ArgumentParser(description='Data Generation')

parser.add_argument('--data_name',
                    type=str,
                    default=None,
                    choices=["AllenCahn_IC_Force_big",
                             "AllenCahn_IC_Force_std",
                             "AllenCahn_IC_Force_small",
                             "AllenCahn_IC_Force_try",
                             "AllenCahn_IC_big",
                             "AllenCahn_IC_std",
                             "AllenCahn_IC_small",
                             "AllenCahn_IC_try",
                             "Burgers_IC_Force_big",
                             "Burgers_IC_Force_std",
                             "Burgers_IC_Force_small",
                             "Burgers_IC_Force_try",
                             "Burgers_IC_big",
                             "Burgers_IC_std",
                             "Burgers_IC_small",
                             "Burgers_IC_try",
                             "AllenCahn_IC_fine_Force_big",
                             "AllenCahn_IC_fine_Force_std",
                             "AllenCahn_IC_fine_Force_small",
                             "AllenCahn_IC_fine_Force_try",
                             "AllenCahn_IC_fine_big",
                             "AllenCahn_IC_fine_std",
                             "AllenCahn_IC_fine_small",
                             "AllenCahn_IC_fine_try",
                             "Burgers_IC_fine_Force_big",
                             "Burgers_IC_fine_Force_std",
                             "Burgers_IC_fine_Force_small",
                             "Burgers_IC_fine_Force_try",
                             "Burgers_IC_fine_big",
                             "Burgers_IC_fine_std",
                             "Burgers_IC_fine_small",
                             "Burgers_IC_fine_try"
                             ],
                    required=True)

args = parser.parse_args()

if __name__ == "__main__":
    if not os.path.exists("./datas/{}_train.py".format(args.data_name)):
        if not os.path.exists("./datas"):
            os.makedirs("./datas")
        print("Generating dataset...")
        data_generation(data_name=args.data_name)
        print("Dataset done.")
    else:
        print("Dataset exists.")
