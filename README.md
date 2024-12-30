# Latent Neural Operator (NeurIPS 2024)

Tian Wang and Chuang Wang. Latent Neural Operator for Solving Forward and Inverse PDE Problem. In Conference on Neural Information Processing Systems (NeurIPS), 2024. [[Paper]](https://arxiv.org/abs/2406.03923)

We evaluate [Latent Neural Operator](https://arxiv.org/abs/2406.03923) using Paddle with six widely used PDE forward problem benchmarks provided by [FNO and GeoFNO](https://github.com/neuraloperator/neuraloperator) and one self-constructed inverse problem inverse benchmark.

**Latent Neural Operator (LNO) reaches the highest precision on four forward problem benchmarks and one inverse problem benchmark, reduces the GPU memory by 50%, speeds up training 1.8 times and keeps the flexibility compared to previous SOTA method. LNO also serve as the best completer and propagator which achieves the most accurate results for solving this inverse problem.**

<p align="center">
<img src=".\assets\LNO.png" height = "300" alt="" align=center />
<br><br>
<b>Figure 1.</b> The overall architecture of Latent Neural Operator.
</p>

<p align="center">
<img src=".\assets\phca.png" height = "300" alt="" align=center />
<br><br>
<b>Figure 2.</b> The working mechanism of Physics-Cross-Attention in encoder and decoder respectively.
</p>


<p align="center">
<img src=".\assets\Forward-1.png" height = "300" alt="" align=center />
<br><br>
<b>Table 1.</b> Prediction error on the six forward problems. The best result is in bold. "∗" means that the results of the method are reproduced by ourselves. "/" means that the method can not handle this benchmark. The column labeled D.C. indicates whether the observation positions and prediction positions are decoupled.
</p>

<p align="center">
<img src=".\assets\Forward-2.png" height = "180" alt="" align=center />
<br><br>
<b>Table 2.</b> Comparison of efficiency between LNO and Transolver on the six forward problem benchmarks. Three metrics are measured: the number of parameters, the cost of GPU memory and the cost of training time per epoch.
</p>

<p align="center">
<img src=".\assets\Inverse-3.png" height = "200" alt="" align=center />
<br><br>
<b>Figure 3.</b> Visualization of Burgers’ equation with different observation situations in different regions. We propose a two-stage strategy. First, we interpolate the solution in the subdomain. Then, we extrapolate from the subdomain to the whole domain.
</p>

<p align="center">
<img src=".\assets\Inverse-1.png" height = "120" alt="" align=center />
<br><br>
<b>Table 3.</b> Relative MAE of different completers in the subdomain in the 1st stage of the inverse
problem with different settings of observation ratio.
</p>

<p align="center">
<img src=".\assets\Inverse-2.png" height = "140" alt="" align=center />
<br><br>
<b>Table 4.</b> The reconstruction error of different propagators in the 2nd stage of the inverse problem.
Propagators are trained to reconstruct the solution in the whole domain based on the ground truth
(G.T.) of the subdomain and are tested using the output of different completers. Relative MAE of
t = 0 and t = 1 is reported.
</p>


<p align="center">
<img src=".\assets\Inverse-4.png" height = "300" alt="" align=center />
<br><br>
<b>Figure 4.</b> Accuracy of solving the inverse problem at different temporal and spatial sampling intervals. (a) Accuracy of LNO as completer. (b) Accuracy of LNO as propagator based on the results of the corresponding completer.
</p>



## Get Started

1. Install Python 3.9 and configure the runtime environment using the following command.

```bash
pip install -r requirements.txt
```

2. Download the dataset from following links into the `./datas` directory.

| Dataset       | Link                                                         |
| ------------- | ------------------------------------------------------------ |
| Darcy         | [[Google Cloud]](https://drive.google.com/drive/folders/1UnbQh2WWc6knEHbLn-ZaXrKUZhp7pjt-) |
| NS2d          | [[Google Cloud]](https://drive.google.com/drive/folders/1UnbQh2WWc6knEHbLn-ZaXrKUZhp7pjt-) |
| AirFoil       | [[Google Cloud]](https://drive.google.com/drive/folders/1YBuaoTdOSr_qzaow-G-iwvbUI7fiUzu8) |
| Elasticity    | [[Google Cloud]](https://drive.google.com/drive/folders/1YBuaoTdOSr_qzaow-G-iwvbUI7fiUzu8) |
| Plasticity    | [[Google Cloud]](https://drive.google.com/drive/folders/1YBuaoTdOSr_qzaow-G-iwvbUI7fiUzu8) |
| Pipe          | [[Google Cloud]](https://drive.google.com/drive/folders/1YBuaoTdOSr_qzaow-G-iwvbUI7fiUzu8) |

3. Train and evaluate model on benchmarks of forward problems using Paddle through the following commands.

```bash 
cp -r datas LNO-Paddle/ForwardProblem/
cd LNO-Paddle/ForwardProblem
bash scripts/LNO_Darcy.sh      # for Darcy
bash scripts/LNO_NS2d.sh       # for NS2d
bash scripts/LNO_Airfoil.sh    # for Airfoil
bash scripts/LNO_ELasticity.sh # for Elasticity
bash scripts/LNO_Plasticity.sh # for Plasticity
bash scripts/LNO_Pipe.sh       # for Pipe
```

4. Train and evaluate model on benchmarks of forward problems using PyTorch through the following commands.

```bash 
cp -r datas LNO-PyTorch/ForwardProblem/
cd LNO-PyTorch/ForwardProblem
bash scripts/LNO_Darcy.sh      # for Darcy
bash scripts/LNO_NS2d.sh       # for NS2d
bash scripts/LNO_Airfoil.sh    # for Airfoil
bash scripts/LNO_ELasticity.sh # for Elasticity
bash scripts/LNO_Plasticity.sh # for Plasticity
bash scripts/LNO_Pipe.sh       # for Pipe
```

5. Train and evaluate model on benchmark of inverse problem using PyTorch through the following commands.

```bash 
cd LNO-PyTorch/InverseProblem
bash scripts/LNO_completer_Burgers.sh     # train completer for Burgers
bash scripts/LNO_propagator_Burgers.sh    # train propagator for Burgers
bash scripts/LNO_Burgers.sh               # infer for Burgers
```

## Citation

If you find this repo useful, please cite our paper. 

```
@inproceedings{wang2024LNO,
  title={Latent Neural Operator for Solving Forward and Inverse PDE Problems},
  author={Tian Wang and Chuang Wang},
  booktitle={Advances in Neural Information Processing},
  year={2024}
}
```

## Contact

If you have any questions or want to use the code, please contact [wangtian2022@ia.ac.cn](mailto:wangtian2022@ia.ac.cn) or [wangchuang@ia.ac.cn](mailto:wangchuang@ia.ac.cn).

## Acknowledgement

We appreciate the following github repos a lot for their valuable code base or datasets:

https://github.com/neuraloperator/neuraloperator

https://github.com/neuraloperator/Geo-FNO

https://github.com/thu-ml/GNOT

https://github.com/thuml/Transolver
