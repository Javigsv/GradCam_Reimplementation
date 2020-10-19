# GradCam_Reimplementation

In this project we replicated the algorithms presented in [*Grad-CAM:
Visual Explanations from Deep Networks via Gradient-based Localization*](https://arxiv.org/pdf/1610.02391.pdf) and [*Grad-CAM++: Improved Visual Explanations for Deep Convolutional Networks*](https://arxiv.org/pdf/1710.11063.pdf). Further extensions and experiments were done. 

The replicated algorithms can be found in `gradCAM.py` and `gradCAMPlusPlus.py`. Experiments were done over ImageNet using VGG16 and ResNet networks (`imageNet_experiments.py`). In these experiments we show the performance of GradCAM and GradCAM++ applied to different convolutions over the network, giving the best explanation when the algorithm is applied to the last convolutional layer as it is stated in the original paper. These experiments' conclusions are result of quantive and qualitative evaluations. 

Another further extension was to try these explanation algorithms (GradCAM and GradCAM++) over audio. For this task a genre classifier that classifies between Folk and Hip-Hop was implemented by taking [*SaewonY repository*](https://github.com/SaewonY/music-genre-classification) as source code. All the code related with the genre classifier can be found in `genreClassifier/`. The data used for the training is [*fma_small_zip*](https://os.unil.cloud.switch.ch/fma/fma_small.zip) and it has been processed as a previous step of the network training: select `Hip-Hop` and `Folk` classes (`genreClassifier/input preprocessing/Take 2 categories.ipynb`) and transform `.wav` files into `.mp3` (`genreClassifier/input preprocessing/convertMp3Wav.ipynb`).

The implementation of the explanation algorithm over audio is in `audio_experiments.py`. 
