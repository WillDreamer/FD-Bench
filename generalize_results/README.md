# Generalize1.csv 

Training data on 2D_CFD_Rand_M0.1_Eta0.01_Zeta0.01_periodic_128_Train.hdf5 on test on different CFD2D Datasets. 

- cfd512: 2D_CFD_Rand_M0.1_Eta1e-08_Zeta1e-08_periodic_512_Train.hdf5
- cfd128_1: 2D_CFD_Rand_M0.1_Eta0.01_Zeta0.01_periodic_128_Train.hdf5
- cfd128_2: 2D_CFD_Rand_M0.1_Eta0.1_Zeta0.1_periodic_128_Train.hdf5
- cfd128_3: 2D_CFD_Rand_M1.0_Eta0.01_Zeta0.01_periodic_128_Train.hdf5
- cfd128_4: 2D_CFD_Rand_M1.0_Eta0.1_Zeta0.1_periodic_128_Train.hdf5


# Generalize2.csv 

Different resolutions (same dataset for train and test):
- cfd2: CFD 256x256 (downscale 2)
- cfd4: CFD 128x128
- cfd8: CFD 64x64
- dr1: Reaction Diffusion 128x128 
- dr2: Reaction Diffusion 64x64 
- dr4: Reaction Diffusion 32x32


# Generalize4.csv

Diffferent scaling (scale 1-5)
- fourier: hidden dim 64, 128, 256, 512, 1024 
- fourier_mode:  mode 24, 32, 40, 48, 56
- conv N_layers: 2,3,4,5,6 
- latent attn_dim: 64, 128, 256, 512, 1024 
- attention hidden_size: 96, 192, 384, 768, 1536
- graph hidden_features: 64, 128, 256, 512, 1024 