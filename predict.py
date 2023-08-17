import os, sys, cv2
import numpy as np
import torch, glob, time
from collections import OrderedDict
from bisenetv2 import BiSeNetV2
from torch.utils.data import DataLoader
from data import LWDDataset

bs = 1
# data
jpgs = glob.glob('/home/hookii/lwd/data/seg0808/*jpg')
jpgs.sort()
n_val=len(jpgs)
# model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model=BiSeNetV2(2, 'eval')
model_path='/home/hookii/lwd/code/bisenet/result/model/4570--0.05221.pth'
paras=torch.load(model_path, map_location='cuda')
dct = OrderedDict()
for key in paras:
	if not 'aux' in key: dct[key]=paras[key]
print(len(dct)) # 383-56=327 
#for key in paras: print(key)
model.load_state_dict(dct)
model.to(device=device)
model.eval()

#palette = np.random.randint(0, 256, (256, 3), dtype=np.uint8)
tm=time.time()
error=0
for jpg in jpgs:
	#im = cv2.imread('/home/lwd/data/cityscapes/leftImg8bit_trainvaltest/leftImg8bit/test/berlin/berlin_000000_000019_leftImg8bit.png')[:, :, ::-1]
	#im = to_tensor(dict(im=im, lb=None))['im'].unsqueeze(0).cuda()
	im = cv2.imread(jpg)
	h,w,c=im.shape
	if h == 720: 
		drop=80
		im=im[drop:]
		h-=drop
	im = im.transpose(2, 0, 1).astype(np.float32).reshape(1,c,h,w)
	im /= 255.0
	im = torch.from_numpy(im).cuda()
	logits = model(im)
	#lb = lb.cuda()
	#loss_pre = criteria_pre(logits, lb)
	#print(loss_pre.item())
	res = logits.argmax(dim=1)
	res = res.squeeze().cpu().numpy().astype('uint8')#.transpose(1,0)
	#res=palette[res]
	res[res>0] = 255
	im=im[0]*255.0
	im = im.permute(1, 2, 0).cpu().numpy().astype('uint8')
	tmp = np.zeros(im.shape).astype('uint8')
	tmp[..., 2] = res/2
	im[res>0] = im[res>0] / 2 + tmp[res>0]
	#cv2.imshow('ss', im)
	#if cv2.waitKey() & 0xff == 27: break
	name = jpg.split("/")[-1]
	print(name)
	cv2.imwrite("result/image/"+name, im)
	png=jpg.replace('jpg', 'png')
	if os.path.exists(png):
		mask = cv2.imread(png, 0)
		mask=(mask!=res).sum()
		error+=mask
tm=time.time()-tm
print(tm/n_val)
print(error/n_val)
