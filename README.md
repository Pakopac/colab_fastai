# Learning Deep Learning Fastai
## Links
Course => https://course.fast.ai/ \
Videos => https://course.fast.ai/videos/ \
Colab notebooks => https://course.fast.ai/start_colab

## Notes
https://github.com/Pakopac/colab_fastai/blob/master/01_intro.ipynb
### "Your first model" part
```
from fastbook import *
```
- fastbook: library for deep learning with all we needed
```
from fastai.vision.all import *
path = untar_data(URLs.PETS)/'images'

def is_cat(x): return x[0].isupper()
dls = ImageDataLoaders.from_name_func(
    path, get_image_files(path), valid_pct=0.2, seed=42,
    label_func=is_cat, item_tfms=Resize(224))

learn = cnn_learner(dls, resnet34, metrics=error_rate)
learn.fine_tune(1)
```
- fastai.vision: image recognition
- untar_data: Get a dataset and untar it
- ImageDataLoaders: Load images for algorithm, valid_pct and seed are used for random, label_func check if is cat and item_tfms transform images
- cnn_learner: build a model with our datas and resnet34 (pre-trained model) using cnn algorithm https://en.wikipedia.org/wiki/Convolutional_neural_network
- fine_tune: train model

```
img = PILImage.create(image_cat())
```
- Used to create an image

```
uploader = widgets.FileUpload()
```
- Upload images and used as an array of images

```
learn.predict(img)
```
- Predict a result with an image

### "Deep Learning Is Not Just for Image Classification" part
--- SegmentationDataLoaders ---
```
path = untar_data(URLs.CAMVID_TINY)
dls = SegmentationDataLoaders.from_label_func(
    path, bs=8, fnames = get_image_files(path/"images"),
    label_func = lambda o: path/'labels'/f'{o.stem}_P{o.suffix}',
    codes = np.loadtxt(path/'codes.txt', dtype=str)
)

learn = unet_learner(dls, resnet34)
learn.fine_tune(8)
```
- SegmentationDataLoaders: reconize specific parts of images
- Codes contain the mapping index to label.
- Using U-Net algorithm https://en.wikipedia.org/wiki/U-Net

--- TextDataLoaders ---
```
from fastai.text.all import *

dls = TextDataLoaders.from_folder(untar_data(URLs.IMDB), valid='test')
learn = text_classifier_learner(dls, AWD_LSTM, drop_mult=0.5, metrics=accuracy)
learn.fine_tune(4, 1e-2)
learn.predict("I really liked that movie!")
```
- Text classifier to predict review of movies

--- TabularDataLoaders ---
- Categories classification 

--- CollabDataLoaders ---
- Recommend system to predict a ranking