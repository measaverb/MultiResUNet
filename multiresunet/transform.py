import torchvision.transforms as transforms
from cv2 import cv2


def resize(im, size):
    im = cv2.resize(im, dsize=(int(size), int(size)), interpolation=cv2.INTER_AREA)
    return im

def preprocessing(image, mask):
    image = image / 255.0
    image, mask = resize(image, 400), resize(mask, 400)
    image_transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )
    
    return image_transform(image).float(), image_transform(mask).float()
