import os
from duckduckgo_search import ddg_images
from fastcore.all import *
from fastdownload import download_url
from fastai.vision.all import *
from setup_logger import logger
from time import sleep


class ImageClassifier:
    def __init__(self, image_categories, *args, **kwargs):
        self.image_categories = image_categories
        self.args = args
        self.kwargs = kwargs

    def search_images(self, term, max_images=30):
        logger.info(f'Searching for "{term}"')
        return L(ddg_images(term, max_results=max_images)).itemgot("image")


if __name__ == "__main__":
    image_categories = ["footbal", "baseball"]
    path = Path("classifier")
    ImageClassifier(image_categories)
