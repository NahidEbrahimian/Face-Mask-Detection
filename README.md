# Face-Mask-Detection

Face Mask Detection on Image and Video using tensorflow and keras

| ![07](https://github.com/NahidEbrahimian/Face-Mask-Detection/assets/82975802/d33df9bd-770a-4120-afd2-125a63a4580a) | ![03](https://github.com/NahidEbrahimian/Face-Mask-Detection/assets/82975802/4bfcad68-166e-4c4b-84e0-ff4994aa3033) |
| :---:         |     :---:      |

| ![01](https://github.com/NahidEbrahimian/Face-Mask-Detection/assets/82975802/ab79f65b-fb75-4abf-a3ae-cbec63bde8b6) | ![02](https://github.com/NahidEbrahimian/Face-Mask-Detection/assets/82975802/b9ff75f1-fdc4-477b-b25b-8c5dcd0bbe2b) |
| :---:         |     :---:      | 



- Train Neural Network on face-mask dataset using tensorflow and keras

- [x] train.ipynb

- [x] inference-img.py

- [x] inference-webcam.py

### Dataset:

Dataset link: [face-mask-dataset](https://www.kaggle.com/datasets/ashishjangra27/face-mask-12k-images-dataset)

#

### Inference:

for inference on image, put your images in `./input` directory and run the following command. output images will be saved on `output` folder.

```
python3 inference-img.py --input input/01.jpeg

```

for inference on video using webcam, run the following command:

```
python3 inference-webcam.py
```

#

### Useful links:

Face-Alignment preprocessing used in the inference step: https://github.com/SajjadAemmi/Face-Alignment

