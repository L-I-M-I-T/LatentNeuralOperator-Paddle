# Latent Neural Operator (NeurIPS 2024)

Tian Wang and Chuang Wang. Latent Neural Operator for Solving Forward and Inverse PDE Problem. In Neural Information Processing Systems (NeurIPS), 2024. [[Paper]](https://arxiv.org/abs/2406.03923)

We evaluate [Latent Neural Operator](https://arxiv.org/abs/2406.03923) using Paddle with six widely used PDE forward problem benchmarks provided by [FNO and GeoFNO](https://github.com/neuraloperator/neuraloperator). The PyTorch version of the code and the experimental scripts related to inverse problems are being organized and will be released soon.

**Latent Neural Operator (LNO) reaches the highest precision on four of these benchmarks, reduces the GPU memory by 50%, speeds up training 1.8 times and keeps the flexibility compared to previous SOTA method.**

<p align="center">
<img src=".\fig\table1.png" height = "300" alt="" align=center />
<br><br>
<b>Table 1.</b> Prediction error on the six forward problems. The best result is in bold. We reproduce the
Transolver by implementing codes independently (marked with ∗) besides the results claimed in
the original paper. The column labeled D.C. indicates whether the observation positions and
prediction positions are decoupled. Standard deviations are computed based on 5 independent trials
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

3. Train and evaluate model on each benchmark using the following commands.

```bash
bash scripts/LNO_Darcy.sh      # for Darcy
bash scripts/LNO_NS2d.sh       # for NS2d
bash scripts/LNO_Airfoil.sh    # for Airfoil
bash scripts/LNO_ELasticity.sh # for Elasticity
bash scripts/LNO_Plasticity.sh # for Plasticity
bash scripts/LNO_Pipe.sh       # for Pipe
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
