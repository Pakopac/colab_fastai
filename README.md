# Learning Deep Learning Fastai
## Links
Course => https://course.fast.ai/ \
Videos => https://course.fast.ai/videos/ \
Colab notebooks => https://course.fast.ai/start_colab

## Notebook #1
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
- ImageDataLoaders: Load images for algorithm, valid_pct is used to avoid overfitting and seed is used for random, label_func check if is cat and item_tfms transform images
- cnn_learner: 
    - build our model using cnn algorithm https://en.wikipedia.org/wiki/Convolutional_neural_network
    - dls: our datas 
    - resnet34: pre-trained model
    - metrics=error_rate: check percent incorretly classified by the model
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
## Notebook #2
https://github.com/Pakopac/colab_fastai/blob/master/02_production.ipynb
### "clean" part
```
if not path.exists():
    path.mkdir()
    for o in bear_types:
        dest = (path/o)
        dest.mkdir(exist_ok=True)
        results = search_images_bing(key, f'{o} bear')
        download_images(dest, urls=results.attrgot('contentUrl'))
```
- Download images for each category
```
failed.map(Path.unlink);
```
 - Delete corrupted images

 ### "From Data to DataLoaders" part

 ```
 bears = DataBlock(
    blocks=(ImageBlock, CategoryBlock), 
    get_items=get_image_files, 
    splitter=RandomSplitter(valid_pct=0.2, seed=42),
    get_y=parent_label,
    item_tfms=Resize(128))
 ```
 - ImageBlock: input data are images, CategoryBlock: labels are categories (grizzly, black, teddy)
 - Splitter: split data into a validation set (randomsplit)
 - Get_y= parent_label: get each item at the name of the parent (ex /black/XXX get each items from /black)

 ```
 dls.valid.show_batch(max_n=4, nrows=1)
 ```
 - Class images into categories

 ```
 interp = ClassificationInterpretation.from_learner(learn)
interp.plot_confusion_matrix()
```
- Confusion matrix: number of actual vs number of predicted