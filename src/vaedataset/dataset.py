from bdataset import InjectDataset
from imgaug.specs import VaeTransform
from imgaug import to_pil
from PIL import Image


class InjectVaeDataset(InjectDataset):
    def __init__(self, size=256, img_key='img_path'):
        super().__init__()
        self.img_key = img_key
        self.trans = VaeTransform(size, resample=Image.BILINEAR)
    
    def transforms(self, idx):
        image = to_pil(self.datas[idx][self.img_key])
        image = image.convert('RGB')
        return self.trans(image)
    

    

        

