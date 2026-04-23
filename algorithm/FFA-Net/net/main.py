import torch,os,sys,torchvision,argparse
import torchvision.transforms as tfs
from metrics import psnr,ssim
from models import *
import time,math
import numpy as np
from torch.backends import cudnn
from torch import optim
import torch,warnings
from torch import nn
#from tensorboardX import SummaryWriter
import torchvision.utils as vutils
warnings.filterwarnings('ignore')
from option import opt,model_name,log_dir
from data_utils import *
from torchvision.models import vgg16
print('log_dir :',log_dir)
print('model_name:',model_name)

models_={
	'ffa':FFA(gps=opt.gps,blocks=opt.blocks),
}
loaders_={
	'its_train':'its_train',
	'its_test':'its_test',
	'ots_train':'ots_train',
	'ots_test':'ots_test',
	'dehaze_train':'dehaze_train',
	'dehaze_test':'dehaze_test'
}
start_time=time.time()
T=opt.steps	
def lr_schedule_cosdecay(t,T,init_lr=opt.lr):
	lr=0.5*(1+math.cos(t*math.pi/T))*init_lr
	return lr

def save_checkpoint(net, optimizer, step, max_psnr, max_ssim, ssims, psnrs, losses, path):
	torch.save({
		'step': step,
		'max_psnr': max_psnr,
		'max_ssim': max_ssim,
		'ssims': ssims,
		'psnrs': psnrs,
		'losses': losses,
		'model': net.state_dict(),
		'optimizer': optimizer.state_dict(),
	}, path)

def load_model_weights(net, checkpoint_path):
	print(f'load pretrained model weights from {checkpoint_path}')
	ckp = torch.load(checkpoint_path, map_location=opt.device, weights_only=False)
	state_dict = ckp['model'] if isinstance(ckp, dict) and 'model' in ckp else ckp
	net.load_state_dict(state_dict, strict=True)

def load_training_state(net, optimizer, checkpoint_path):
	print(f'resume from {checkpoint_path}')
	ckp = torch.load(checkpoint_path, map_location=opt.device, weights_only=False)
	net.load_state_dict(ckp['model'])
	optimizer.load_state_dict(ckp['optimizer'])
	return (
		ckp.get('losses', []),
		ckp.get('step', 0),
		ckp.get('max_ssim', 0),
		ckp.get('max_psnr', 0),
		ckp.get('ssims', []),
		ckp.get('psnrs', []),
	)

def train(net,loader_train,loader_test,optim,criterion):
	losses=[]
	start_step=0
	max_ssim=0
	max_psnr=0
	ssims=[]
	psnrs=[]
	if opt.resume_checkpoint:
		losses,start_step,max_ssim,max_psnr,ssims,psnrs = load_training_state(net, optim, opt.resume_checkpoint)
		print(f'start_step:{start_step} start training ---')
	elif opt.resume and os.path.exists(opt.last_checkpoint):
		losses,start_step,max_ssim,max_psnr,ssims,psnrs = load_training_state(net, optim, opt.last_checkpoint)
		print(f'start_step:{start_step} start training ---')
	elif opt.pretrained_checkpoint:
		load_model_weights(net, opt.pretrained_checkpoint)
		print('start finetuning from pretrained weights *** ')
	else:
		print('train from scratch *** ')
	train_iter = iter(loader_train)
	for step in range(start_step+1,opt.steps+1):
		net.train()
		lr=opt.lr
		if not opt.no_lr_sche:
			lr=lr_schedule_cosdecay(step,T)
			for param_group in optim.param_groups:
				param_group["lr"] = lr  
		try:
			x,y=next(train_iter)
		except StopIteration:
			train_iter = iter(loader_train)
			x,y=next(train_iter)
		x=x.to(opt.device);y=y.to(opt.device)
		out=net(x)
		loss=criterion[0](out,y)
		if opt.perloss:
			loss2=criterion[1](out,y)
			loss=loss+0.04*loss2
		
		loss.backward()
		
		optim.step()
		optim.zero_grad()
		losses.append(loss.item())
		print(f'\rtrain loss : {loss.item():.5f}| step :{step}/{opt.steps}|lr :{lr :.7f} |time_used :{(time.time()-start_time)/60 :.1f}',end='',flush=True)

		#with SummaryWriter(logdir=log_dir,comment=log_dir) as writer:
		#	writer.add_scalar('data/loss',loss,step)

		if step % opt.eval_step ==0 :
			with torch.no_grad():
				ssim_eval,psnr_eval=test(net,loader_test, max_psnr,max_ssim,step)

			print(f'\nstep :{step} |ssim:{ssim_eval:.4f}| psnr:{psnr_eval:.4f}')

			# with SummaryWriter(logdir=log_dir,comment=log_dir) as writer:
			# 	writer.add_scalar('data/ssim',ssim_eval,step)
			# 	writer.add_scalar('data/psnr',psnr_eval,step)
			# 	writer.add_scalars('group',{
			# 		'ssim':ssim_eval,
			# 		'psnr':psnr_eval,
			# 		'loss':loss
			# 	},step)
			ssims.append(ssim_eval)
			psnrs.append(psnr_eval)
			if ssim_eval > max_ssim and psnr_eval > max_psnr :
				max_ssim=max(max_ssim,ssim_eval)
				max_psnr=max(max_psnr,psnr_eval)
				best_path = os.path.join(opt.output_dir, f'model-{step}.pt')
				save_checkpoint(net, optim, step, max_psnr, max_ssim, ssims, psnrs, losses, best_path)
				print(f'\n model saved at step :{step}| max_psnr:{max_psnr:.4f}|max_ssim:{max_ssim:.4f}')

	save_checkpoint(net, optim, step, max_psnr, max_ssim, ssims, psnrs, losses, opt.last_checkpoint)
	print(f'\n last model saved at :{opt.last_checkpoint}')

	np.save(f'./numpy_files/{model_name}_{opt.steps}_losses.npy',losses)
	np.save(f'./numpy_files/{model_name}_{opt.steps}_ssims.npy',ssims)
	np.save(f'./numpy_files/{model_name}_{opt.steps}_psnrs.npy',psnrs)

def test(net,loader_test,max_psnr,max_ssim,step):
	net.eval()
	torch.cuda.empty_cache()
	ssims=[]
	psnrs=[]
	total = len(loader_test)
	start_eval = time.time()
	print(f'\nstart validation at step {step}, total_images={total}', flush=True)
	#s=True
	for i ,(inputs,targets) in enumerate(loader_test):
		inputs=inputs.to(opt.device);targets=targets.to(opt.device)
		pred=net(inputs)
		# # print(pred)
		# tfs.ToPILImage()(torch.squeeze(targets.cpu())).save('111.png')
		# vutils.save_image(targets.cpu(),'target.png')
		# vutils.save_image(pred.cpu(),'pred.png')
		ssim1=ssim(pred,targets).item()
		psnr1=psnr(pred,targets)
		ssims.append(ssim1)
		psnrs.append(psnr1)
		if ((i + 1) % opt.eval_print_freq == 0) or (i + 1 == total):
			elapsed = time.time() - start_eval
			print(
				f'validation progress: {i + 1}/{total} | '
				f'current_psnr:{psnr1:.4f} | current_ssim:{ssim1:.4f} | '
				f'time_used:{elapsed/60:.1f} min',
				flush=True
			)
		#if (psnr1>max_psnr or ssim1 > max_ssim) and s :
		#		ts=vutils.make_grid([torch.squeeze(inputs.cpu()),torch.squeeze(targets.cpu()),torch.squeeze(pred.clamp(0,1).cpu())])
		#		vutils.save_image(ts,f'samples/{model_name}/{step}_{psnr1:.4}_{ssim1:.4}.png')
		#		s=False
	return np.mean(ssims) ,np.mean(psnrs)


if __name__ == "__main__":
	loader_train=build_loader(loaders_[opt.trainset])
	loader_test=build_loader(loaders_[opt.testset])
	net=models_[opt.net]
	net=net.to(opt.device)
	if opt.device=='cuda':
		net=torch.nn.DataParallel(net)
		cudnn.benchmark=True
	criterion = []
	criterion.append(nn.L1Loss().to(opt.device))
	if opt.perloss:
			vgg_model = vgg16(pretrained=True).features[:16]
			vgg_model = vgg_model.to(opt.device)
			for param in vgg_model.parameters():
				param.requires_grad = False
			criterion.append(PerLoss(vgg_model).to(opt.device))
	optimizer = optim.Adam(params=filter(lambda x: x.requires_grad, net.parameters()),lr=opt.lr, betas = (0.9, 0.999), eps=1e-08)
	optimizer.zero_grad()
	train(net,loader_train,loader_test,optimizer,criterion)
	
