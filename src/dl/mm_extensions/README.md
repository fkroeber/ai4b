## MMdet & MMrotate extensions for AI4B

The three modules contain customised code required to reproduce the deep learning instance segmentation results.
They extend the functionality provided by mmdetection & mmrotate by...
* tailoring the dataloading to the AI4B dataset 
* speeding up the evaluation (fastevalcoco)
* extending to provided models (new backbones, detectors, heads)

The code was developed using the following package versions:

>mmdetection: 3.0.0 </br>
>mmrotate : 1.0.0rc1 </br>
>mmcv: 2.0.0 </br>
>mmengine: 0.7.2 </br>
>torch version: 2.0.0 </br>


To reproduce the analyses, one needs to...</br>
... install mmdetection and mmrotate as they are not contained in the conda environment file (`env.yaml`)   
... put the current directory on the main level of the mmdetection/mmrotate repositories (for all models not based on mmrotate you may use the mmdetection library only, remove any pieces of code referring to mmrotate and change the registration of the classes to mmdetection) </br>
... additionally disable the assertion regarding the number of channels in `mmengine/model/base_model/data_preprocessor.py` to allow multi-band inputs with more than 3 channels</br> 

For an exemplary config file making a call to custom_imports to integrate the modules, find the provided configuration for the best model [here](https://1drv.ms/f/s!AlhSFEgs5NDDjbFrgzFhbG732wujog?e=9G7zrb). 