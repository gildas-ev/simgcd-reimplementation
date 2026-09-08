from PIL import Image

import cifar

if __name__ == "__main__":
    subclass = cifar.CIFAR100WithIndex()
    img = subclass.__getitem__(0)[0]
    img.show()
    print("flag")
