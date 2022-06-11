# SALAI-Net

SALAI-Net is a package for species-agnostic Local Ancestry Inference (or ancestry 
deconvolution). In other words, you can perform LAI on any species or set of 
ancestries, given a reference panel of single-ancestry individuals.

![Template matching](Base_model_topk.drawio.png)

Fig. 1: Novel template matching block.


## Community

## Installation
### Native Linux
1. Clone the repo
   ```git clone https://github.com/AI-sandbox/SALAI-Net.git```
2. Install dependencies
```pip install -r requirements```. Check [pytorch.org]() for installation of the appropiate version of pytorch and torchvision for your system.
### Docker
## Usage
### Download data and pretrained models.

Data for the main results is available in 
https://drive.google.com/file/d/1GHZvlvdAUoFDYOUfFKhTtTHCCy8INZSM/view?usp=sharing

Main model (for 1000G and dogs)
and model for the Hapmap dataset are in
https://drive.google.com/file/d/1FG67JzMq_1GhtLHnmnRxRSIy66Vhkg_K/view?usp=sharing


### Perform LAI



```python src/SALAI.py [args]```
Arguments:
- ```--model-cp```: Path to ```.pt``` checkpoint of the model's paramters
- ```--model-args```: (Optional) Path to ```.pckl``` file containing the arguments used to instanciate the model. By default take the ones in the checkpoint's parent folder.
- ```--query, -q```: Path to vcf files containing the query sequences.
- ```--reference, -r```: Path to vcf file containing the reference haplotype data.
- ```--map, -m```: Path to sample map indicating the ancestry of each reference.
- ```--out-folder, -o```: Folder where the predictions will be dumped.
- ```--batch-size, -b```: Number of references predicted at the same time, for the purpose of controlling memory consumption at inference.

usage example:

    python src/SALAI.py \
        --model-cp models/main_model/models/best_model.pth \
        -q published_data/dogs/128gen/vcf.vcf \
        -r published_data/dogs/chr30_expt1_filtered.vcf.gz \
        -m published_data/dogs/128gen/ref_panel/train2.map \
        -o prediction_folder

For whole genome data use ```main_model/models/best_model.pth``` and for shorter sequences like hapmap use ```hapmap_model/models/best_model.pth```

The code runs by default on GPU if it is available, otherwise it runs on CPU. To run on CPU when GPU is available, deactivate GPU usage by running ```export CUDA_VISIBLE_DEVICES=''``` before running SALAI

### Fine-tune SALAI-Net

## Cite



