import torch
from torchvision import models
from thop import profile
from torchsummary import summary
from torchstat import stat
from ptflops import get_model_complexity_info
model = models.vgg16()
input = torch.randn(1, 3, 224, 224)
flops, params = profile(model, inputs=(input, ))
print(flops)
print(params)
# summary(model,input_size=( 3, 224, 224))
stat(model, (3, 224, 224))

# with torch.cuda.device(0):
#   net = models.vgg11()
#   macs, params = get_model_complexity_info(net, (3, 224, 224), as_strings=True,
#                                            print_per_layer_stat=True, verbose=True)
#   print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
#   print('{:<30}  {:<8}'.format('Number of parameters: ', params))