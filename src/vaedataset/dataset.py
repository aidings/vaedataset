import torch
from bdataset import InjectDataset, InjectBucketDataset, ImageBuckets, BuckNode
from imgaug.specs import VaeTransform
from PIL import Image


class InjectVaeDataset(InjectDataset):
    def __init__(self, size=256, img_key='img_path'):
        super().__init__()
        self.img_key = img_key
        self.trans = VaeTransform(size, resample=Image.BILINEAR)
    
    def transforms(self, idx):
        image = Image.open(self.datas[idx][self.img_key])
        image = image.convert('RGB')
        return self.trans(image)


class InjectVaeBucketDataset(InjectBucketDataset):
    def __init__(self, buckets: ImageBuckets, img_key='img_path'):
        super().__init__(buckets)
        self.img_key = img_key
        self.trans = VaeTransform()
    
    def data2node(self, line_data):
        width, height = line_data['img_size']
        idx = len(self.datas)
        return BuckNode(width, height, idx)
    
    def transforms(self, idx, resolution):
        image = Image.open(self.datas[idx][self.img_key])
        image = image.convert('RGB')
        image = image.resize(resolution, Image.BILINEAR)
        return self.trans(image) 

    def totensor(self, datas):
        return torch.stack(datas, dim=0)


    

        

