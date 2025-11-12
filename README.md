## RIA

A novel multi-omics framework is proposed, titled Robust Integrative Analysis of Multi-omics Datasets via Nuclear-norm Maximization.

## License

Part of the code is derived from PRAGA (Huang et al. 2025).

## Dependencies

- python==3.8.6
- anndata==0.9.2
- numpy==1.24.4
- pandas==2.0.3
- rpy2==3.4.5
- scanpy==1.9.8
- scikit-learn==1.3.2
- scikit-misc==0.2.0
- scipy==1.10.1
- torch==2.4.1

The packages listed above are the main packages used in the experiments. Most PyTorch 2.0+ environments can run the experiments directly.

## Data

Please download Human Lymph Node dataset (Long et al. 2024) and spatial epigenome–transcriptome mouse brain dataset (Zhang et al. 2023) from https://zenodo.org/records/14591305, and unzip them into `./Data/`.

## Quick start



We create a Python 3.8 environment named ria:

```
conda create -n ria python=3.8
conda activate ria
```

Install packages:

```
- python==3.8.6
- anndata==0.9.2
- numpy==1.24.4
- pandas==2.0.3
- rpy2==3.4.5
- scanpy==1.9.8
- scikit-learn==1.3.2
- scikit-misc==0.2.0
- scipy==1.10.1
- torch==2.4.1
```

Run the script after installation.

```
sh run.sh
```

The quantification results and visualizations will be saved in the `./results`.