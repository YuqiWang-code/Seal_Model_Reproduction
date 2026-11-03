## Long Range Diffusion for Weakly Camouflaged Object Segmentation
This project provides the code and results for 'Long Range Diffusion for Weakly Camouflaged Object Segmentation'
> Authors:
> Rui Wang,
> Caijuan Shi, 
> Weixiang Gao,
> Changyu Duan,
> Ao Cai,
> Fei Yu,
> Yunchao Wei

### Network Architecture
![image](Images/image1.png)

### Preparation
The training and testing experiments are conducted using PyTorch with a single GeForce RTX 1080Ti GPU of 12 GB Memory.
#### Configuring your environment:
* Creating a virtual environment : `conda create -n LRDNet python=3.9`
* Installing necessary packages: `pip install -r requirements.txt`
#### Downloading Training and Testing Sets
* Download train datasets S-COD(COD10K-train+CAMO-train): [TrainDatasets](https://drive.google.com/file/d/1D9bf1KeeCJsxxri6d2qAC7z6O1X_fxpt/view?usp=sharing). 
"1" stands for foregrounds, "2" for backgrounds, and "0" for unlabeled regions.
* Download test datasets (CAMO-test+COD10K-test-test+NC4K ):[TestDatasets](https://drive.google.com/file/d/1QEGnP9O7HbN_2tH999O3HRIsErIVYalx/view?usp=sharing)
#### Pretrained Backbone Model
* Download pretrained backbone model:[ResNet50](https://pan.baidu.com/s/1ywGmDdpMfz8_XD0EXEpxrw?pwd=52m1), and put it in `./pth`
### Training 
The source-domain training and Seal cross-domain testing protocol is described below.

### Source scribble training and Seal cross-domain test

The original 4,040-image training set contains full masks in `TrainDataset/GT`; use the corresponding [Scribble-COD annotations](https://drive.google.com/file/d/1u7PRtZDu2vXCRe0o2SplVYa7ESoZQFR/view) in `data_original/Scribble` for scribble supervision. Place the original images in `data_original/TrainDataset/Imgs` and the ImageNet-1K ResNet50 weights in `pth/resnet50-0676ba61.pth`.

The completed source-domain run is available on RSML-3. Reuse `experiments/SCOD/stage2/models/100_epoch.pth` and `experiments/SCOD/test_maps/`. Both stages end at the predefined epoch 100 without Seal validation. Logs and maps are under `experiments/SCOD/`. The exact protocol, eight metrics, and archived experiments are in `../docs/LRDNet_Seal实验记录.md`. `run_scod_source.sh` is the source-only training recipe for cases requiring source retraining.

### Archived Seal adaptation
Seal box training and Seal validation checkpoint selection are disabled. Historical artifacts remain on the server and are documented separately from the source-domain result.
### Testing Configuration
* Seal cross-domain testing: `python infer_seal.py --stage 2 --checkpoint experiments/SCOD/stage2/models/100_epoch.pth --experiment_root experiments/SCOD`.
* We provide LRDNet [testing maps](https://pan.baidu.com/s/1eddSG534wDik5vbvwXAbqA?pwd=4xnq) and [training weights](https://pan.baidu.com/s/1V9Zk6QLbIB1xCv8-sDUFiw?pwd=ponf) presented in the papers.
### Evaluation
* Tools: [PySODMetrics](https://github.com/lartpang/PySODMetrics) A simple and efficient implementation of SOD metrics.
### Results
![image](Images/image2.png)

![image](Images/image3.png)

### Credit
The code is partly based on [CRNet](https://github.com/dddraxxx/Weakly-Supervised-Camouflaged-Object-Detection-with-Scribble-Annotations).




