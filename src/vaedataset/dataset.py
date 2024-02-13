from bdataset import InjectImageDataset
from imgaug.specs import VaeTransform
from PIL import Image


class InjectVaeDataset(InjectImageDataset):
    def __init__(self, size=256, img_key='img_path'):
        super().__init__(img_key)
        self.trans = VaeTransform(size, resample=Image.BILINEAR)
    
    def transforms(self, idx):
        image = self.imread(idx)
        return self.trans(image)
    

    

        

