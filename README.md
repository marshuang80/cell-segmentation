# Cell Segmentation

PyTorch implementation of several neural network segmentaion models (UNet, FusionNet, DialatedConvolution) for cell image segmentation. The trained models from this repository are used for the segmentation plugin [segmentify](https://github.com/transformify-plugins/segmentify) for [Napari](https://github.com/napari/napari)

| Original | Segmentation |
| --- | --- |
| ![](figs/original.png) | ![](figs/segmentation.png) |

## Dataset

The following datasets were used to train and test the different cell segmentatoin models:

[Nuclei Dataset](https://www.kaggle.com/c/data-science-bowl-2018/overview)
[HPA Dataset](https://www.kaggle.com/c/human-protein-atlas-image-classification)
[Nuero Dataset](http://neurofinder.codeneuro.org/)


## Process Data

After downloading the dataset from the links above, each dataset can be parsed and converted to a HDF5 file using scripts from *./process_data*. For example:

```python ./process_data/hpa_create_hdf5.py --input_dir PATH_TO_DATA --output_dir PATH_TO_OUTPUT```



