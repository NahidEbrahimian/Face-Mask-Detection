# Face-Mask-Detection

Face Mask Detection on Image and Video using tensorflow and keras

| ![01](https://user-images.githubusercontent.com/82975802/136102598-225cee41-fe9b-4150-99cd-f8b5945768de.jpg) | ![02](https://user-images.githubusercontent.com/82975802/136102619-d0370afc-21df-4bd7-9338-9b98981ec99d.jpg) | 
| :---:         |     :---:      | 
| ![077](https://user-images.githubusercontent.com/82975802/136111552-86fe9f32-9fa1-49fc-9eca-b6b2c434765a.jpg)| ![033](https://user-images.githubusercontent.com/82975802/136111585-43a23985-962d-4c3e-a53b-eb4ce0a52f48.jpg)|


- Train Neural Network on face-mask dataset using tensorflow and keras

- [x] train.ipynb

- [x] inference-img.py

- [x] inference-wb.py

### Dataset:

Dataset link: [face-mask-dataset]( ashishjangra27/face-mask-12k-images-dataset)

#

### Inference:

for inference on image, put your images in `./input` directory and run the following command. output images will be saved on `output` folder.

```
python3 inference-img.py --input input/01.jpeg

```

for inference on video using webcam, run the following command:

#

### Useful links:

Face-Alignment preprocessing used in the inference step: https://github.com/SajjadAemmi/Face-Alignment
