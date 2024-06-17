## Requirements
This code requires the following:
* Python==3.9
* Pytorch==1.11.0
* Pytorch Geometric==2.0.4
* Numpy==1.21.2
* Scikit-learn==1.0.2
* OGB==1.3.3
* NetworkX==2.7.1
* FAISS-GPU==1.7.2

## Usage

Just run the script corresponding to the experiment and dataset you want. For instance:

Run out-of-distribution detection on AIDS (ID) and DHFR (OOD) datasets:
```
unzip data.zip
bash script/run_AIDS+DHFR.sh
bash script/run_ogbg-molesol+ogbg-molmuv.sh
bash script/run_ogbg-molfreesolv+ogbg-moltoxcast.sh
bash script/run_ogbg-moltox21+ogbg-molsider.sh
...
```
