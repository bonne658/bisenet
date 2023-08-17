import torch, sys
from collections import OrderedDict
from bisenetv2 import BiSeNetV2

net=BiSeNetV2(2, 'eval')
paras = torch.load('result/model/4570--0.05221.pth', map_location="cpu")
net.eval()
dct = OrderedDict()
for key in paras:
	if not 'aux' in key: dct[key]=paras[key]
#sys.exit()
net.load_state_dict(dct)
x = torch.randn((1, 3, 640, 1280))
traced_script_module = torch.jit.trace(net, x)
traced_script_module.save('bis0723.pt')
