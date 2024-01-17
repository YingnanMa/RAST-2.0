# -*- coding: utf-8 -*-


import argparse
from pathlib import Path
import os
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.utils.data as data
from PIL import Image
from PIL import ImageFile
from tensorboardX import SummaryWriter
from torchvision import transforms
from tqdm import tqdm
from torchvision.utils import save_image
from matplotlib import pyplot as plt
import matplotlib.colors as mcolors
import sys
sys.path.append('/content/drive/Shareddrives/Style-Transfer/IEContraAST-main/')
import net
from sampler import InfiniteSamplerWrapper
import numpy as np
import pandas as pd
from pandas import DataFrame

cudnn.benchmark = True
Image.MAX_IMAGE_PIXELS = None  # Disable DecompressionBombError
ImageFile.LOAD_TRUNCATED_IMAGES = True  # Disable OSError: image file is truncated


def train_transform():
    transform_list = [
        transforms.Resize(size=(512, 512)),
        transforms.RandomCrop(256),
        transforms.ToTensor()
    ]
    return transforms.Compose(transform_list)


class FlatFolderDataset(data.Dataset):
    def __init__(self, root, transform):
        super(FlatFolderDataset, self).__init__()
        self.root = root
        self.paths = os.listdir(self.root)
        self.transform = transform

    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(os.path.join(self.root, path)).convert('RGB')
        img = self.transform(img)
        return img

    def __len__(self):
        return len(self.paths)

    def name(self):
        return 'FlatFolderDataset'


def adjust_learning_rate(optimizer, iteration_count):
    """Imitating the original implementation"""
    lr = args.lr / (1.0 + args.lr_decay * iteration_count)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


break_point=80000
parser = argparse.ArgumentParser()
# Basic options
parser.add_argument('--log_dir', default='./logs',
                    help='Directory to save the log')   
parser.add_argument('--content_dir', type=str, default='./train2014',
                    help='Directory path to a batch of content images')
parser.add_argument('--style_dir', type=str, default='./train',
                    help='Directory path to a batch of style images')
parser.add_argument('--save_dir', default='./experiments_round36',
                    help='Directory to save the model')
parser.add_argument('--sample_path', type=str, default='./samples_round36',
                     help='Directory to save the intermediate samples')
parser.add_argument('--vgg', type=str,default='./model/vgg_normalised.pth',
                     help='Directory to load the encoder model')
parser.add_argument('--breakpoint_transformer', type=str, default='./experiments_round36/transformer_iter_' + str(break_point) + '.pth', 
                    help='Directory to load the breakpoint transformer model')
parser.add_argument('--breakpoint_decoder', type=str, default='./experiments_round36/decoder_iter_' + str(break_point) + '.pth', 
                    help='Directory to load the breakpoint decoder model')
parser.add_argument('--breakpoint_optimizer', type=str, default='./experiments_round36/optimizer_iter_' + str(break_point) + '.pth', 
                    help='Directory to load the breakpoint optimizer model')
parser.add_argument('--breakpoint_optimizer_D1', type=str, default='./experiments_round36/optimizer_D1_iter_' + str(break_point) + '.pth', 
                    help='Directory to load the breakpoint optimizer D1 model')
parser.add_argument('--breakpoint_optimizer_D2', type=str, default='./experiments_round36/optimizer_D2_iter_' + str(break_point) + '.pth', 
                    help='Directory to load the breakpoint optimizer D2 model')

# training options
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--lr_decay', type=float, default=5e-5)
parser.add_argument('--max_iter', type=int, default=160100)
parser.add_argument('--batch_size', type=int, default=8)
#parser.add_argument('--transitive_weight', type=float, default=2.0)
#parser.add_argument('--diff_weight', type=float, default=2.0)
#parser.add_argument('--restoration_weight', type=float, default=1.0)
parser.add_argument('--contrastive_weight_c', type=float, default=1.0)
parser.add_argument('--contrastive_weight_s', type=float, default=1.0)
parser.add_argument('--gan_weight', type=float, default=5.0)
parser.add_argument('--n_threads', type=int, default=16)
parser.add_argument('--save_model_interval', type=int, default=5000)
parser.add_argument('--start_iter', type=float, default=0)
parser.add_argument('--BK', type=bool, default=False)
parser.add_argument('--content_weight', type=float, default=0.25)
parser.add_argument('--style_weight', type=float, default=0.25)
parser.add_argument('--content_transitive_weight', type=float, default=0.5)
parser.add_argument('--style_transitive_weight', type=float, default=0.5)
#parser.add_argument('--content_diff_weight', type=float, default=1.0)
parser.add_argument('--style_diff_weight', type=float, default=0.5)
parser.add_argument('--content_restoration_weight', type=float, default=0.25)
parser.add_argument('--style_restoration_weight', type=float, default=0.25)
args = parser.parse_args('')

#os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device('cuda')
save_dir = Path(args.save_dir)
save_dir.mkdir(exist_ok=True, parents=True)
log_dir = Path(args.log_dir)
log_dir.mkdir(exist_ok=True, parents=True)
writer = SummaryWriter(log_dir=str(log_dir))

decoder = net.decoder
vgg = net.vgg

valid = 1
fake = 0
D1 = net.MultiDiscriminator()
D1.to(device)

D2 = net.MultiDiscriminator()
D2.to(device)

vgg.load_state_dict(torch.load(args.vgg))
vgg = nn.Sequential(*list(vgg.children())[:44])
network = net.Net(vgg, decoder, args.start_iter, args.breakpoint_transformer, args.breakpoint_decoder)
network.train()
network.to(device)

content_tf = train_transform()
style_tf = train_transform()
content_dataset = FlatFolderDataset(args.content_dir, content_tf)
style_dataset = FlatFolderDataset(args.style_dir, style_tf)

content_iter = iter(data.DataLoader(
    content_dataset, batch_size=int(args.batch_size / 2),
    sampler=InfiniteSamplerWrapper(content_dataset),
    num_workers=args.n_threads))
style_iter = iter(data.DataLoader(
    style_dataset, batch_size=int(args.batch_size / 2),
    sampler=InfiniteSamplerWrapper(style_dataset),
    num_workers=args.n_threads))

optimizer = torch.optim.Adam([{'params': network.decoder.parameters()},
                              {'params': network.transform.parameters()},
                              {'params': network.proj_style.parameters()},
                              {'params': network.proj_content.parameters()}], lr=args.lr)
optimizer_D1 = torch.optim.Adam(D1.parameters(), lr=args.lr, betas=(0.5, 0.999))
optimizer_D2 = torch.optim.Adam(D2.parameters(), lr=args.lr, betas=(0.5, 0.999))

if(args.start_iter > 0):
    optimizer.load_state_dict(torch.load(args.breakpoint_optimizer))
    optimizer_D1.load_state_dict(torch.load(args.breakpoint_optimizer_D1))
    optimizer_D2.load_state_dict(torch.load(args.breakpoint_optimizer_D2))
    
#Draw loss pic====================================
def draw_plt(loss_data,loss_name,color):
    plt.rcParams.update({"font.size":14})
    plt.xlabel("Iterations",fontsize=14)
    plt.ylabel('Value',fontsize=14)
    x_axis=[i*1000 for i in range(3,len(loss_data)+1)]
    y_axis=loss_data[2:]
    plt.plot(x_axis,y_axis,c=color,label=str(loss_name))
    
def draw_pic(loss_lst1,loss_lst2,loss_lst4,loss_lst5,loss_lst6,loss_lst7):
    plt.figure(figsize=(25,6))
    plt.rcParams.update({"font.size":14})
    plt.subplot(1,3,1)
    plt.title("Transitive Loss and Style Difference Loss",fontsize=14)
    draw_plt(loss_lst1,"Content Transitive Loss","xkcd:blue")
    draw_plt(loss_lst2, "Style Transitive Loss", "xkcd:aquamarine")
    draw_plt(loss_lst4, "Style Difference Loss", "xkcd:goldenrod")
    plt.legend(fontsize=14)
    

    plt.subplot(1,3,2)
    plt.title("Restoration Loss",fontsize=14)
    draw_plt(loss_lst5,"Content Restoration Loss", "xkcd:purpley")
    draw_plt(loss_lst6, "Style Restoration Loss", "xkcd:bubblegum")
    plt.legend(fontsize=14)
    
    plt.subplot(1,3,3)
    plt.title("Total Loss",fontsize=14)
    draw_plt(loss_lst7, "Total Loss","xkcd:coral")
    plt.legend(fontsize=14)
    plt.savefig("./Loss-V36.png")

    

def mean_loss(lst):
    new_lst=[]
    i=1
    while i <len(lst):
        point=np.mean(lst[i:i+1000])
        new_lst.append(point)
        i+=1000
    return new_lst



#====================================================
filename = './loss_v36.xlsx'
columns=["content_transitive","style_transitive","style_difference","content_restoration","style_restoration","loss_identity1",
        "loss_identity2","loss_contrastive_c","loss_contrastive_s","loss_gan_g1","loss_gan_d1","loss_gan_g2","loss_gan_d2", "content_loss", "style_loss",
        "total_loss"]

if args.BK==True:
    df_loss = pd.read_excel(filename)[:args.start_iter]
    print("breakpoint xlsx length:",len(df_loss))
    content_transitive_loss_lst=df_loss["content_transitive"].to_list()
    style_transitive_loss_lst=df_loss["style_transitive"].to_list()
    style_diff_loss_lst=df_loss["style_difference"].to_list()
    content_restoration_loss_lst=df_loss["content_restoration"].to_list()
    style_restoration_loss_lst=df_loss["style_restoration"].to_list()
    loss_identity1_loss_lst=df_loss["loss_identity1"].to_list()
    loss_identity2_loss_lst=df_loss["loss_identity2"].to_list()
    loss_contrastive_c_loss_lst=df_loss["loss_contrastive_c"].to_list()
    loss_contrastive_s_loss_lst=df_loss["loss_contrastive_s"].to_list()
    loss_gan_g1_loss_lst=df_loss["loss_gan_g1"].to_list()
    loss_gan_d1_loss_lst=df_loss["loss_gan_d1"].to_list()
    loss_gan_g2_loss_lst=df_loss["loss_gan_g2"].to_list()
    loss_gan_d2_loss_lst=df_loss["loss_gan_d2"].to_list()
    content_loss_lst=df_loss["content_loss"].to_list()
    style_loss_lst=df_loss["style_loss"].to_list()
    total_loss_lst=df_loss["total_loss"].to_list()
else:
    content_transitive_loss_lst=[]
    style_transitive_loss_lst=[]
    style_diff_loss_lst=[]
    content_restoration_loss_lst=[]
    style_restoration_loss_lst=[]
    loss_identity1_loss_lst=[]
    loss_identity2_loss_lst=[]
    loss_contrastive_c_loss_lst=[]
    loss_contrastive_s_loss_lst=[]
    loss_gan_g1_loss_lst=[]
    loss_gan_d1_loss_lst=[]
    loss_gan_g2_loss_lst=[]
    loss_gan_d2_loss_lst=[]
    content_loss_lst=[]
    style_loss_lst=[]
    total_loss_lst=[]

for i in tqdm(range(args.start_iter, args.max_iter)):
    adjust_learning_rate(optimizer, iteration_count=i)
    adjust_learning_rate(optimizer_D1, iteration_count=i)
    adjust_learning_rate(optimizer_D2, iteration_count=i)
    content_images = next(content_iter).to(device)
    style_images = next(style_iter).to(device)
    

    ######################################################
    content_images_ = content_images[1:]
    content_images_ = torch.cat([content_images_, content_images[0:1]], 0)
    content_images = torch.cat([content_images, content_images_], 0)
    style_images = torch.cat([style_images, style_images], 0)

    ######################################################

    output1, rc1, rs1, output2, rc2, rs2, l_identity1, l_identity2, loss_contrastive_c, loss_contrastive_s, content_transitive_loss, style_transitive_loss, style_diff_loss, content_restoration_loss, style_restoration_loss, content_loss, style_loss= network(content_images, style_images, args.batch_size)

    # train discriminator
    loss_gan_d1 = D1.compute_loss(style_images, valid) + D1.compute_loss(output1.detach(), fake)
    optimizer_D1.zero_grad()
    loss_gan_d1.backward()
    optimizer_D1.step()
    
    loss_gan_d2 = D2.compute_loss(content_images, valid) + D2.compute_loss(output2.detach(), fake)
    optimizer_D2.zero_grad()
    loss_gan_d2.backward()
    optimizer_D2.step()

    # train generator
    content_transitive_loss = args.content_transitive_weight * content_transitive_loss
    style_transitive_loss = args.style_transitive_weight * style_transitive_loss
    #content_diff_loss = args.diff_weight * content_diff_loss
    style_diff_loss = args.style_diff_weight * style_diff_loss
    content_restoration_loss = args.content_restoration_weight * content_restoration_loss
    style_restoration_loss = args.style_restoration_weight * style_restoration_loss
    loss_c = args.content_weight * content_loss
    loss_s = args.style_weight * style_loss
    loss_contrastive_c = args.contrastive_weight_c * loss_contrastive_c
    loss_contrastive_s = args.contrastive_weight_s * loss_contrastive_s
    loss_gan_g1 = args.gan_weight * D1.compute_loss(output1, valid)
    loss_gan_g2 = args.gan_weight * D2.compute_loss(output2, valid)
    loss1 =l_identity1 * 50 + l_identity2 * 1 + loss_contrastive_c + loss_contrastive_s + loss_gan_g1+ loss_gan_g2
    loss = content_transitive_loss + style_transitive_loss + style_diff_loss + content_restoration_loss + style_restoration_loss + loss1 + loss_c + loss_s


    content_transitive_loss_lst.append(content_transitive_loss.item())
    style_transitive_loss_lst.append(style_transitive_loss.item())
    #content_diff_loss_value= 1/content_diff_loss.item()
    #content_diff_loss_lst.append(content_diff_loss_value)
    style_diff_loss_value= 1/style_diff_loss.item()
    style_diff_loss_lst.append(style_diff_loss_value)
    content_restoration_loss_lst.append(content_restoration_loss.item())
    style_restoration_loss_lst.append(style_restoration_loss.item())

    loss_identity1_loss_lst.append(l_identity1.item())
    loss_identity2_loss_lst.append(l_identity2.item())
    loss_contrastive_c_loss_lst.append(loss_contrastive_c.item())
    loss_contrastive_s_loss_lst.append(loss_contrastive_s.item())
    loss_gan_g1_loss_lst.append(loss_gan_g1.item())
    loss_gan_d1_loss_lst.append(loss_gan_d1.item())
    loss_gan_g2_loss_lst.append(loss_gan_g2.item())
    loss_gan_d2_loss_lst.append(loss_gan_d2.item())
    content_loss_lst.append(loss_c.item())
    style_loss_lst.append(loss_s.item())
    total_loss_lst.append(loss.item())

    


    if i>0 and (i+1)%1000==0:
        n_content_transitive_loss=mean_loss(content_transitive_loss_lst)
        n_style_transitive_loss=mean_loss(style_transitive_loss_lst)
        #n_content_diff_loss=mean_loss(content_diff_loss_lst)
        n_style_diff_loss=mean_loss(style_diff_loss_lst)
        n_content_restoration_loss=mean_loss(content_restoration_loss_lst)
        n_style_restoration_loss=mean_loss(style_restoration_loss_lst)
        n_total=mean_loss(total_loss_lst)
        
        if i>2999:
            draw_pic(n_content_transitive_loss,n_style_transitive_loss,n_style_diff_loss,n_content_restoration_loss,n_style_restoration_loss,n_total)
        
        if (i+1) % 5000 == 0:
                all_loss = []
                for j in range(len(total_loss_lst)):
                    all_loss.append([content_transitive_loss_lst[j], style_transitive_loss_lst[j], style_diff_loss_lst[j],
                                     content_restoration_loss_lst[j],
                                     style_restoration_loss_lst[j], loss_identity1_loss_lst[j], loss_identity2_loss_lst[j],
                                     loss_contrastive_c_loss_lst[j],
                                     loss_contrastive_s_loss_lst[j], loss_gan_g1_loss_lst[j], loss_gan_d1_loss_lst[j],
                                     loss_gan_g2_loss_lst[j], loss_gan_d2_loss_lst[j], content_loss_lst[j], style_loss_lst[j],
                                     total_loss_lst[j]])
    
                df_1 = pd.DataFrame(all_loss, columns=columns)
                print("len of saved xlsx:",len(df_1))
                df_1.to_excel(filename, index=False)

 

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    writer.add_scalar('loss_content_transitive', content_transitive_loss.item(), i + 1)
    writer.add_scalar('loss_style_transitive', style_transitive_loss.item(), i + 1)
    #writer.add_scalar('loss_content_diff', content_diff_loss.item(), i + 1)
    writer.add_scalar('loss_style_diff', style_diff_loss.item(), i + 1)
    writer.add_scalar('loss_content_restoration', content_restoration_loss.item(), i + 1)
    writer.add_scalar('loss_style_restoration', style_restoration_loss.item(), i + 1)
    #writer.add_scalar('loss_content', loss_c.item(), i + 1)
    #writer.add_scalar('loss_style', loss_s.item(), i + 1)
    writer.add_scalar('loss_identity1', l_identity1.item(), i + 1)
    writer.add_scalar('loss_identity2', l_identity2.item(), i + 1)
    writer.add_scalar('loss_contrastive_c', loss_contrastive_c.item(), i + 1)  # attention
    writer.add_scalar('loss_contrastive_s', loss_contrastive_s.item(), i + 1)  # attention
    writer.add_scalar('loss_gan_g1', loss_gan_g1.item(), i + 1)  # attention
    writer.add_scalar('loss_gan_d1', loss_gan_d1.item(), i + 1)
    writer.add_scalar('loss_gan_g2', loss_gan_g2.item(), i + 1)  # attention
    writer.add_scalar('loss_gan_d2', loss_gan_d2.item(), i + 1)

    writer.add_scalar('loss_content', loss_c.item(), i + 1)
    writer.add_scalar('loss_style', loss_s.item(), i + 1)
    
    ############################################################################
    output_dir = Path(args.sample_path)
    output_dir.mkdir(exist_ok=True, parents=True)
    if (i == 0) or ((i + 1) % 500 == 0):
        output = torch.cat([style_images, content_images, output1, output2, rc1, rc2, rs1, rs2], 2)
        output_name = output_dir / 'output{:d}.jpg'.format(i + 1)
        save_image(output, str(output_name))
        #save_image(output, str(output_name), args.batch_size)
    ############################################################################

    if (i + 1) % args.save_model_interval == 0 or (i + 1) == args.max_iter:
        state_dict = decoder.state_dict()
        for key in state_dict.keys():
            state_dict[key] = state_dict[key].to(torch.device('cpu'))
        torch.save(state_dict,
                   '{:s}/decoder_iter_{:d}.pth'.format(args.save_dir,
                                                           i + 1))
        state_dict = network.transform.state_dict()
        for key in state_dict.keys():
            state_dict[key] = state_dict[key].to(torch.device('cpu'))
        torch.save(state_dict,
                   '{:s}/transformer_iter_{:d}.pth'.format(args.save_dir,
                                                           i + 1))
        state_dict = optimizer.state_dict()
        torch.save(state_dict,
                   '{:s}/optimizer_iter_{:d}.pth'.format(args.save_dir,
                                                           i + 1))
        state_dict = optimizer_D1.state_dict()
        torch.save(state_dict,
                   '{:s}/optimizer_D1_iter_{:d}.pth'.format(args.save_dir,
                                                           i + 1))   
                                                           
        state_dict = optimizer_D2.state_dict()
        torch.save(state_dict,
                   '{:s}/optimizer_D2_iter_{:d}.pth'.format(args.save_dir,
                                                           i + 1))                                      
                                



                                             
writer.close()
