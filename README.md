# VDW-GNNs: Vector diffusion wavelets for geometric graph neural networks

A PyTorch/PyTorch-Geometric framework for building (optionally) rotationally-equivariant graph neural networks that utilize diffusion wavelets for both scalar and vector node features.

---
## 1&nbsp;&nbsp;Required packages
Core dependencies:
- python>=3.11
- pytorch
- torch-geometric
- torch-scatter, torch-cluster, torch-sparse[1]
- torchmetrics
- accelerate
- numpy
- scikit-learn
- scipy
- h5py
- pyyaml

Optional dependencies:
- e3nn (for Tensor Field Networks models, etc.)
- wandb
- pandas
- matplotlib
- pot [used by MARBLE]
- cebra
- statannotations

[1] Can be installed as a dependency of pytorch-geometric. See its [installation guide](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html).

An example installation inside a conda environment:
```
mamba install -c pytorch -c nvidia -c conda-forge python=3.11 pytorch torchvision torchaudio pytorch-cuda=12.4 numpy scikit-learn pandas matplotlib h5py pyyaml -y && \
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric --extra-index-url https://data.pyg.org/whl/torch-$(python -c "import torch; print(torch.__version__.split('+')[0])") && \
pip install torchmetrics accelerate
```

For wind data experiments:
```mamba install -c conda-forge xarray netCDF4``` [for reading '.nc' data files]

For macaque reaching experiments (baselines):
```pip install cebra```
```mamba install pot``` [used by MARBLE]

For results plots:
```mamba install -c conda-forge statannotations``` [will also install seaborn, statsmodels, etc.]

---
## 2&nbsp;&nbsp;Datasets

### 2.1 Ellipsoids
These are synthetic datasets that can be easily reproduced with relevant code in the `data_processing` folder of this repo.

Example: generate an ellipsoids dataset (also computes the $\mathbf{P}$ and $\mathbf{Q}$ operators for VDW-GNN models):
```bash
python3 {ROOT}/scripts/python/generate_ellipsoid_dataset.py \
  --save_dir /path/to/data/ellipsoids \
  --config {ROOT}/config/yaml_files/ellipsoids/experiment.yaml \
  --pq_h5_name pq_tensor_data_512.h5 \
  --random_seed 457892 \
  --num_samples 512 \
  --num_nodes_per_graph 128 \
  --knn_graph_k 5 \
  --abc_means 3.0 1.0 1.0 \
  --abc_stdevs 0.5 0.2 0.2 \
  --local_pca_kernel_fn gaussian \
  --num_oversample_points 1024 \
  --k_laplacian 10 \
  --laplacian_type sym_norm \
  --dirac_types max min \
  --random_harmonic_k 16 \
  --random_harmonic_coeff_bounds 1.0 2.0 \
  --modulation_scale 0.9
```

### 2.2 Earth surface wind velocity, 1 January 2016
Downloaded from the NOAA Physical Science Lab data repo: https://psl.noaa.gov/data/gridded/data.ncep.reanalysis.html (accessed December 2025).

You will need to download the u- and v- wind measurements separately from this site. First, identify the data set with these attributes:

| Variable         | Statistic | Level | TimeScale |
|------------------|-----------|-------|-----------|
| u-wind \| v-wind | Mean      | 10 m  | Daily     |

Then, for each, click the data set's 'Plot/subset all files' icon (which looks like a data plot) in the Options column of the data set's row. This opens a data set explorer page, where you can subset to the single date, and then click 'Download subset of data defined by options' to download an `.nc` file. For a single day, these are not large files (~100 kilobytes).

For use with this repo, save the downloaded u- and v-wind files at `{ROOT}/data/`, under the filenames `u-wind_1Jan2016_mean_10m.nc` and `v-wind_1Jan2016_mean_10m.nc`.


### 2.3 Multi-channel neural recordings
Data files for these experiments can be accessed from the MARBLE (Gosztolai et al. 2025) data repo:
- Trial data: https://dataverse.harvard.edu/api/access/datafile/6969883
- Kinematics (target) data: https://dataverse.harvard.edu/api/access/datafile/6969885
- Trial ids: https://dataverse.harvard.edu/api/access/datafile/6963200


---
## 3&nbsp;&nbsp;Running experiments

### Jupyter notebook
The `run_experiments.ipynb` notebook contains instructions and script call examples for running our experiments and aggregating results.

### SLURM scripts
We also share our SLURM scheduler scripts, in `scripts/slurm/`. Note that these scripts assume a conda environment: be sure to modify their directory and CONDA_ENV variables, etc., appropriately to your cluster settings before running.

---
## 4&nbsp;&nbsp;Note: experiment configuration precedence
If you wish to alter experiment settings, note that in this repo, configuration values are layered in the following order (highest priority first):
1. Command-line arguments
2. Model-specific YAML (i.e., the file passed to a `--config` argument)
3. Experiment YAML in the same directory as the model YAML files for a given experiment/dataset (`experiment.yaml`)
4. Default values in `config/*_config.py` files or function signatures, etc.