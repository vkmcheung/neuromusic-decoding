# Decoding drums, instrumentals, vocals, and mixed sources in music using human brain activity with fMRI
by Vincent K.M. Cheung (SonyCSL), Lana Okuma (RIKEN), Kazuhisa Shibata (RIKEN), Kosetsu Tsukuda (AIST), Masataka Goto (AIST), Shinichi Furuya (SonyCSL)

Check out our ISMIR 2023 paper [here](https://archives.ismir.net/ismir2023/paper/000022.pdf)


<code>*data.npy</code> files contain numpy arrays of BOLD activations in the respective areas as subjects listen to each instrumental source

<code>*fname.npy</code> files contain corresponding information about subject, instrumental source, and song-id, separated by underscores

<code>run_4-way_decoding.py</code> performs leave-one-subject-out cross-validation in decoding the four instrumental sources from fMRI data using CNN, random forest, and SVM classifiers

<code>run_2-way_decoding.py</code> performs leave-one-subject-out cross-validation in decoding the presence of drums, vocals, or instrumentals in the sound stimuli from fMRI data using CNN, random forest, and SVM classifiers

## Requirements:
- Python 3.9
- Tensorflow/Keras 2.10+ (?)
- sklearn 1.2+

## Supplementary Information:
coming soon
