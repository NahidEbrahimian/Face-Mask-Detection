# Face-Mask-Detection

Face Mask Detection on Image and Video using tensorflow and keras

| ![0777](https://user-images.githubusercontent.com/82975802/136189978-6f784a4a-a977-4783-915b-b4ad7dedc1f5.jpg)| ![0333](https://user-images.githubusercontent.com/82975802/136189989-2567c476-9fcd-4a78-97ab-e3b8d0fde874.jpg)| ![facedetection](https://user-images.githubusercontent.com/82975802/136190127-570a26b3-9778-497d-a269-738858fe6521.gif)|
| :---:         |     :---:      |      :---:      |

| ![01](https://user-images.githubusercontent.com/82975802/136102598-225cee41-fe9b-4150-99cd-f8b5945768de.jpg) | ![02](https://user-images.githubusercontent.com/82975802/136102619-d0370afc-21df-4bd7-9338-9b98981ec99d.jpg) | 
| :---:         |     :---:      | 



- Train Neural Network on face-mask dataset using tensorflow and keras

- [x] train.ipynb

- [x] inference-img.py

- [x] inference-webcam.py

### Dataset:

Dataset link: [face-mask-dataset]( ashishjangra27/face-mask-12k-images-dataset)

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

