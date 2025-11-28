from fcmae_model import convnextv2_pico
from torchinfo import summary
import torch

#use pico
models = convnextv2_pico(img_size=196).cuda()
x = torch.randn([1,3,196,196]).cuda()
print(models(x)[0],models(x)[1].shape,models(x)[2].shape)
out = summary(models, (1, 3, 196,196))