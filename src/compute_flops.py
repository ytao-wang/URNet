from torchstat import stat
from importlib import import_module
from option import args
from model.edsr import EDSR
from model.rcan import RCAN
# from model.mynet import make_model
from model.urn import make_model

# model = EDSR(args)
model = make_model(args)
stat(model, (3, 256, 256))

# # 统计网络层数与参数
# net = model
# print(net)
# params = list(net.parameters())
# print(len(params))
# k=0
# for i in params:
#     l =1
#     #print("该层的结构："+ str(list(i.size())))
#     for j in i.size():
#         l *= j
#     # print("该层参数和："+ str(l))
#     k = k+l
# print('总参数和：'+ str(k))

