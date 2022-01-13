## Reproducing Neural Discrete Representation Learning
### Course Project for [IFT 6135 - Representation Learning](https://ift6135h18.wordpress.com/)

Project Report link: [final_project.pdf](final_project.pdf)
### Download the dataset
```
cd data/miniimagenet
gdown --id 1pQK7CDStL4Pvzf4AlMNWcYcwS0D-3pJa
unzip mini.zip
rm mini.zip
cd ../..
```

### Download the miniimagenet vqvae model k=64
```
gdown --id 1UGlBPd7U5nBloHDbYRMtbH2x5Zf2FjVa
```

### Check the results of this trained vqvae 
See loadvqvaek64.ipynb

### Run it yourself
https://colab.research.google.com/drive/1BH2RK088d5-w-H4oSrs4t5zJwLxctRXV?usp=sharing

### Instructions
1. To train the VQVAE with default arguments as discussed in the report, execute:
```
python vqvae.py --data-folder /tmp/miniimagenet --output-folder models/vqvae
```
2. To train the PixelCNN prior on the latents, execute:
```
python pixelcnn_prior.py --data-folder /tmp/miniimagenet --model models/vqvae --output-folder models/pixelcnn_prior
```