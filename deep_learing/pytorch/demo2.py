import torch, torchvision
# model = torchvision.models.vgg
model = torchvision.models.vgg16()
# model = model.to('cuda:0')
from torchsummary import summary
summary(model, (3, 224, 224),device="cpu")

