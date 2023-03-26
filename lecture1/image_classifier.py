#import os
#from fastcore.all import *
from fastai.vision.all import *
from image_downloader import ImageDownloader


class ImageClassifier:
    def __init__(self, image_categories, download=True, **kwargs):
        self.image_categories = image_categories
        if download == True:
            [ImageDownloader(image, **kwargs).download_images() for image in image_categories]

    def learn(self, dls, n_epochs=3, pretrained=resnet18, metrics=error_rate, **kwargs):
        learn = vision_learner(dls, pretrained, metrics)
        learn.fine_tune(n_epochs)
        self.learn = learn
        if 'download_model' in kwargs:
            learn.export('model.pkl')
        return learn
    


if __name__ == "__main__":
    image_categories = ["football", "baseball", "basketball"]
    path_folder = "classifier_sports"
    n_epochs = 6
    bs = 32
    pretrained = resnet18


    ic = ImageClassifier(image_categories,  download=True, path_folder=path_folder)
    dls = DataBlock(
        blocks=(ImageBlock, CategoryBlock),
        get_items=get_image_files,
        splitter=RandomSplitter(valid_pct=0.2, seed=42),
        get_y=parent_label,
        item_tfms=[Resize(192, method='squish')]
    ).dataloaders(Path(path_folder), bs=bs)
    learn  = ic.learn(dls=dls, n_epochs=n_epochs, pretrained=pretrained, download_model=True)
    print(learn.predict('sports_images/football.jpg'))
    print(learn.predict('sports_images/baseball.jpg'))
    print(learn.predict('sports_images/basketball.jpg'))
    