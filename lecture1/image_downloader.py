import os
import time
from duckduckgo_search import ddg_images
from fastai.vision.all import Path, L, download_images, resize_images
from setup_logger import logger


class ImageDownloader:
    def __init__(self, image, path_folder=None):
        self.image = image
        path = "classifier" if path_folder == None else path_folder
        self.path = os.path.join(os.getcwd(), path)
        print(self.path)

    def search_images(self, additional_term=""):
        logger.info(f'Searching for "{self.image}"')
        return L(ddg_images(self.image + additional_term, max_results=30)).itemgot(
            "image"
        )

    def download_images(self):
        dest = Path(self.path) / self.image
        dest.mkdir(exist_ok=True, parents=True)
        download_images(dest, urls=self.search_images("photo"))
        time.sleep(10)
        download_images(dest, urls=self.search_images())
        resize_images(
            Path(self.path) / self.image,
            max_size=400,
            dest=Path(self.path) / self.image,
        )


if __name__ == "__main__":
    id = ImageDownloader("real madrid", path_folder="classifier_sports")
    id.download_images()
