import os
import time
from duckduckgo_search import ddg_images
from fastai.vision.all import Path, L, download_images, resize_images, verify_images, get_image_files
#from setup_logger import logger


class ImageDownloader:
    def __init__(self, image, path_folder=None):
        self.image = image
        self.image_name = "_".join(image.split())
        path = "classifier" if path_folder == None else path_folder
        self.path = os.path.join(os.getcwd(), path)

    def search_images(self, additional_term=""):
        #logger.info(f'Searching for "{self.image}"')
        return L(ddg_images(self.image + additional_term, max_results=30)).itemgot(
            "image"
        )

    def download_images(self, fail_check=True):
        dest = Path(self.path) / self.image_name
        dest.mkdir(exist_ok=True, parents=True)
        download_images(dest, urls=self.search_images("photo"))
        time.sleep(10)
        download_images(dest, urls=self.search_images())
        resize_images(
            Path(self.path) / self.image_name,
            max_size=400,
            dest=Path(self.path) / self.image_name,
        )
        if fail_check == True:
            n_fail = verify_images(get_image_files(Path(self.path))).map(Path.unlink)
            #logger.info(f"Number of failed downloads {len(n_fail)}")

    def download_unique_image(self):
        download_url(search_images(self.image, max_images=1)[0], f'{self.image}.jpg', show_progress=False)



if __name__ == "__main__":
    id = ImageDownloader("real madrid", path_folder="classifier_sports")
    id.download_images()
