# SALAI-Net

SALAI-Net is a package for species-agnostic Local Ancestry Inference (or ancestry 
deconvolution). In other words, you can perform LAI on any species or set of 
ancestries, given a reference panel of single-ancestry individuals.

## Community

## Install
### Native Linux
1. Clone the repo
   ```git clone []```
2. Install dependencies
3. 
### Docker
## Usage
### Perform LAI on your data

```python src/SALAI.py [args]```
Arguments:
- ```--model-cp```: Path to ```.pt``` checkpoint of the model's paramters
- ```--model-args```: (Optional) Path to ```.pckl``` file containing the arguments used to instanciate the model. By default take the ones in the checkpoint's parent folder.
- ```--query, -q```: Path to vcf files containing the query sequences.
- ```--reference, -r```: Path to vcf file containing the reference haplotype data.
- ```--map, -m```: Path to sample map indicating the ancestry of each reference.
- ```--out-folder, -o```: Folder where the predictions will be dumped.
- ```--batch-size, -b```: Number of references predicted at the same time, for the purpose of controlling memory consumption at inference.

### Fine-tune SALAI-Net

## Cite



