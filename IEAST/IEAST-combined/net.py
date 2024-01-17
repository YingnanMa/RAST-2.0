# -*- coding: utf-8 -*-

import numpy as np
import torch
import torch.nn as nn
import cv2


def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std

def mean_variance_norm(feat):
    size = feat.size()
    mean, std = calc_mean_std(feat)
    normalized_feat = (feat - mean.expand(size)) / std.expand(size)
    return normalized_feat

def _calc_feat_flatten_mean_std(feat):
    # takes 3D feat (C, H, W), return mean and std of array within channels
    assert (feat.size()[0] == 3)
    assert (isinstance(feat, torch.FloatTensor))
    feat_flatten = feat.view(3, -1)
    mean = feat_flatten.mean(dim=-1, keepdim=True)
    std = feat_flatten.std(dim=-1, keepdim=True)
    return feat_flatten, mean, std

decoder = nn.Sequential(
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 256, (3, 3)),
    nn.ReLU(),
    nn.Upsample(scale_factor=2, mode='nearest'),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 128, (3, 3)),
    nn.ReLU(),
    nn.Upsample(scale_factor=2, mode='nearest'),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 128, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 64, (3, 3)),
    nn.ReLU(),
    nn.Upsample(scale_factor=2, mode='nearest'),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 64, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 3, (3, 3)),
)

vgg = nn.Sequential(
    nn.Conv2d(3, 3, (1, 1)),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(3, 64, (3, 3)),
    nn.ReLU(),  # relu1-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 64, (3, 3)),
    nn.ReLU(),  # relu1-2
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 128, (3, 3)),
    nn.ReLU(),  # relu2-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 128, (3, 3)),
    nn.ReLU(),  # relu2-2
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 256, (3, 3)),
    nn.ReLU(),  # relu3-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-4
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 512, (3, 3)),
    nn.ReLU(),  # relu4-1, this is the last layer used
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-4
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU()  # relu5-4
)

projection_style = nn.Sequential(
    nn.Linear(in_features=256, out_features=128),
    nn.ReLU(),
    nn.Linear(in_features=128, out_features=128)
)

projection_content = nn.Sequential(
    nn.Linear(in_features=512, out_features=256),
    nn.ReLU(),
    nn.Linear(in_features=256, out_features=128)
)


class MultiDiscriminator(nn.Module):
    def __init__(self, in_channels=3):
        super(MultiDiscriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, normalize=True):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        # Extracts three discriminator models
        self.models = nn.ModuleList()
        for i in range(3):
            self.models.add_module(
                "disc_%d" % i,
                nn.Sequential(
                    *discriminator_block(in_channels, 64, normalize=False),
                    *discriminator_block(64, 128),
                    *discriminator_block(128, 256),
                    *discriminator_block(256, 512),
                    nn.Conv2d(512, 1, 3, padding=1)
                ),
            )

        self.downsample = nn.AvgPool2d(in_channels, stride=2, padding=[1, 1], count_include_pad=False)

    def compute_loss(self, x, gt):
        """Computes the MSE between model output and scalar gt"""
        loss = sum([torch.mean((out - gt) ** 2) for out in self.forward(x)])
        return loss

    def forward(self, x):
        outputs = []
        for m in self.models:
            outputs.append(m(x))
            x = self.downsample(x)
        return outputs


class SANet(nn.Module):
    def __init__(self, in_planes):
        super(SANet, self).__init__()
        self.f = nn.Conv2d(in_planes, in_planes, (1, 1))
        self.g = nn.Conv2d(in_planes, in_planes, (1, 1))
        self.h = nn.Conv2d(in_planes, in_planes, (1, 1))
        self.sm = nn.Softmax(dim = -1)
        self.out_conv = nn.Conv2d(in_planes, in_planes, (1, 1))
    def forward(self, content, style):
        F = self.f(mean_variance_norm(content))
        G = self.g(mean_variance_norm(style))
        H = self.h(style)
        b, c, h, w = F.size()
        F = F.view(b, -1, w * h).permute(0, 2, 1)
        b, c, h, w = G.size()
        G = G.view(b, -1, w * h)
        S = torch.bmm(F, G)
        S = self.sm(S)
        b, c, h, w = H.size()
        H = H.view(b, -1, w * h)
        O = torch.bmm(H, S.permute(0, 2, 1))
        b, c, h, w = content.size()
        O = O.view(b, c, h, w)
        O = self.out_conv(O)
        O += content
        return O

class Transform(nn.Module):
    def __init__(self, in_planes):
        super(Transform, self).__init__()
        self.sanet4_1 = SANet(in_planes = in_planes)
        self.sanet5_1 = SANet(in_planes = in_planes)
        #self.upsample5_1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.merge_conv_pad = nn.ReflectionPad2d((1, 1, 1, 1))
        self.merge_conv = nn.Conv2d(in_planes, in_planes, (3, 3))
    def forward(self, content4_1, style4_1, content5_1, style5_1):
        self.upsample5_1 = nn.Upsample(size=(content4_1.size()[2], content4_1.size()[3]), mode='nearest')
        return self.merge_conv(self.merge_conv_pad(self.sanet4_1(content4_1, style4_1) + self.upsample5_1(self.sanet5_1(content5_1, style5_1))))

class Net(nn.Module):
    def __init__(self, encoder, decoder, start_iter, breakpoint_transformer, breakpoint_decoder):
        super(Net, self).__init__()
        enc_layers = list(encoder.children())
        self.enc_1 = nn.Sequential(*enc_layers[:4])  # input -> relu1_1
        self.enc_2 = nn.Sequential(*enc_layers[4:11])  # relu1_1 -> relu2_1
        self.enc_3 = nn.Sequential(*enc_layers[11:18])  # relu2_1 -> relu3_1
        self.enc_4 = nn.Sequential(*enc_layers[18:31])  # relu3_1 -> relu4_1
        self.enc_5 = nn.Sequential(*enc_layers[31:44])  # relu4_1 -> relu5_1

        #projection
        self.proj_style = projection_style
        self.proj_content = projection_content

        #transform
        self.transform = Transform(in_planes = 512)
        self.decoder = decoder
        self.cross_entropy_loss = nn.CrossEntropyLoss()

        if(start_iter > 0):
            self.transform.load_state_dict(torch.load(breakpoint_transformer))
            self.decoder.load_state_dict(torch.load(breakpoint_decoder))
        self.mse_loss = nn.MSELoss()
        
        # fix the encoder
        for name in ['enc_1', 'enc_2', 'enc_3', 'enc_4', 'enc_5']:
            for param in getattr(self, name).parameters():
                param.requires_grad = False

    # extract relu1_1, relu2_1, relu3_1, relu4_1, relu5_1 from input image
    def encode_with_intermediate(self, input):
        results = [input]
        for i in range(5):
            func = getattr(self, 'enc_{:d}'.format(i + 1))
            results.append(func(results[-1]))
        return results[1:]
    


    def calc_content_loss(self, input, target, norm = False):
        if(norm == False):
          return self.mse_loss(input, target)
        else:
          return self.mse_loss(mean_variance_norm(input), mean_variance_norm(target))


    def calc_style_loss(self, input, target):
        input_mean, input_std = calc_mean_std(input)
        target_mean, target_std = calc_mean_std(target)
        return self.mse_loss(input_mean, target_mean) + \
               self.mse_loss(input_std, target_std)
    
    

    def compute_contrastive_loss(self, feat_q, feat_k, tau, index):
        out = torch.mm(feat_q, feat_k.transpose(1, 0)) / tau
        #loss = self.cross_entropy_loss(out, torch.zeros(out.size(0), dtype=torch.long, device=feat_q.device))
        loss = self.cross_entropy_loss(out, torch.tensor([index], dtype=torch.long, device=feat_q.device))
        return loss

    def style_feature_contrastive(self, input):
        # out = self.enc_style(input)
        out = torch.sum(input, dim=[2, 3])
        out = self.proj_style(out)
        out = out / torch.norm(out, p=2, dim=1, keepdim=True)
        return out

    def content_feature_contrastive(self, input):
        #out = self.enc_content(input)
        out = torch.sum(input, dim=[2, 3])
        out = self.proj_content(out)
        out = out / torch.norm(out, p=2, dim=1, keepdim=True)
        return out
    
    def forward(self, content, style, batch_size):
        content_feats = self.encode_with_intermediate(content)
        style_feats = self.encode_with_intermediate(style)
        
        #restoration right
        stylized1 = self.transform(content_feats[3], style_feats[3], content_feats[4], style_feats[4])
        output1 = self.decoder(stylized1)
        opt1_feats = self.encode_with_intermediate(output1)
        
        stylized_rc1 = self.transform(opt1_feats[3], content_feats[3], opt1_feats[4], content_feats[4])
        rc1 = self.decoder(stylized_rc1)
        rc1_feats = self.encode_with_intermediate(rc1)
        
        stylized_rs1 = self.transform(style_feats[3], opt1_feats[3], style_feats[4], opt1_feats[4])
        rs1 = self.decoder(stylized_rs1)
        rs1_feats = self.encode_with_intermediate(rs1)
        
        #restoration left
        stylized2 = self.transform(style_feats[3], content_feats[3], style_feats[4], content_feats[4])
        output2 = self.decoder(stylized2)
        opt2_feats = self.encode_with_intermediate(output2)
        
        stylized_rc2 = self.transform(content_feats[3], opt2_feats[3], content_feats[4], opt2_feats[4])
        rc2 = self.decoder(stylized_rc2)
        rc2_feats = self.encode_with_intermediate(rc2)
        
        stylized_rs2 = self.transform(opt2_feats[3], style_feats[3], opt2_feats[4], style_feats[4])
        rs2 = self.decoder(stylized_rs2)
        rs2_feats = self.encode_with_intermediate(rs2)
        
        #restoration loss functions right
        content_transitive_loss1 = self.calc_content_loss(rc1_feats[3], content_feats[3], norm = True) + self.calc_content_loss(rc1_feats[4], content_feats[4], norm = True)

        style_diff_loss1 =self.calc_style_loss(opt1_feats[0], content_feats[0])
        for i in range(1, 5):
            style_diff_loss1 += self.calc_style_loss(opt1_feats[i], content_feats[i])
        style_diff_loss1=1/style_diff_loss1
        
        style_transitive_loss1 =self.calc_style_loss(rs1_feats[0], style_feats[0])
        for i in range(1, 5):
            style_transitive_loss1 +=self.calc_style_loss(rs1_feats[i], style_feats[i])
        
        #content_diff_loss1 = self.calc_content_loss(opt1_feats[0], style_feats[0], norm = True) 
        #for i in range(1, 5):
        #    content_diff_loss1 += self.calc_content_loss(opt1_feats[i], style_feats[i], norm = True)
        #content_diff_loss1 = 1/content_diff_loss1
        
        #restoration loss functions left
        content_transitive_loss2 = self.calc_content_loss(rs2_feats[3], style_feats[3], norm = True) + self.calc_content_loss(rs2_feats[4], style_feats[4], norm = True)

        style_diff_loss2 = self.calc_style_loss(opt2_feats[0], style_feats[0])
        for i in range(1, 5):
            style_diff_loss2 += self.calc_style_loss(opt2_feats[i], style_feats[i])
        style_diff_loss2=1/style_diff_loss2
        
        style_transitive_loss2 = self.calc_style_loss(rc2_feats[0], content_feats[0])
        for i in range(1, 5):
            style_transitive_loss2 += self.calc_style_loss(rc2_feats[i], content_feats[i])
        
        #content_diff_loss2 = self.calc_content_loss(opt2_feats[0], content_feats[0], norm = True)
        #for i in range(1, 5):
        #    content_diff_loss2 += self.calc_content_loss(opt2_feats[i], content_feats[i])
        #content_diff_loss2 = 1/content_diff_loss2

        #restoration loss
        content_restoration_loss = self.calc_content_loss(rc1_feats[3], rc2_feats[3], norm = True) + self.calc_content_loss(rc1_feats[4], rc2_feats[4], norm = True) + self.calc_content_loss(rs1_feats[3], rs2_feats[3], norm = True) + self.calc_content_loss(rs1_feats[4], rs2_feats[4], norm = True)
        #content_restoration_loss = self.calc_content_loss(rc1_feats[0], rc2_feats[0], norm = True)
        #for i in range(1, 5):
            #content_restoration_loss += self.calc_content_loss(rc1_feats[i], rc2_feats[i], norm = True)
            
        #content_restoration_loss += self.calc_content_loss(rs1_feats[0], rs2_feats[0], norm = True)
        #for i in range(1, 5):
            #content_restoration_loss += self.calc_content_loss(rs1_feats[i], rs2_feats[i], norm = True)
        
        
        style_restoration_loss = self.calc_style_loss(rc1_feats[0], rc2_feats[0])
        for i in range(1, 5):
            style_restoration_loss += self.calc_style_loss(rc1_feats[i], rc2_feats[i])
            
        style_restoration_loss += self.calc_style_loss(rs1_feats[0], rs2_feats[0])
        for j in range(1, 5):
            style_restoration_loss += self.calc_style_loss(rs1_feats[j], rs2_feats[j])
            
        content_transitive_loss = content_transitive_loss1 + content_transitive_loss2
        style_transitive_loss = style_transitive_loss1 + style_transitive_loss2
        #content_diff_loss = content_diff_loss1 + content_diff_loss2
        style_diff_loss = style_diff_loss1 + style_diff_loss2
        
        # add content and style loss
        content_loss = self.calc_content_loss(opt1_feats[3], content_feats[3], norm=True) + self.calc_content_loss(opt1_feats[4], content_feats[4], norm=True) + self.calc_content_loss(opt2_feats[3], style_feats[3], norm=True) + self.calc_content_loss(opt2_feats[4], style_feats[4], norm=True)
        style_loss = self.calc_style_loss(opt1_feats[0], style_feats[0])
        for p in range(1,5):
            style_loss += self.calc_style_loss(opt1_feats[p], style_feats[p])

        style_loss += self.calc_style_loss(opt2_feats[0], content_feats[0])
        for q in range(1,5):
            style_loss += self.calc_style_loss(opt2_feats[q], content_feats[q])
        #original content loss and style loss
        #loss_c = self.calc_content_loss(g_t_feats[3], content_feats[3], norm = True) + self.calc_content_loss(g_t_feats[4], content_feats[4], norm = True)
        #loss_s = self.calc_style_loss(g_t_feats[0], style_feats[0])
        #for i in range(1, 5):
        #    loss_s += self.calc_style_loss(g_t_feats[i], style_feats[i])


        """IDENTITY LOSSES"""
        Icc = self.decoder(self.transform(content_feats[3], content_feats[3], content_feats[4], content_feats[4]))
        Iss = self.decoder(self.transform(style_feats[3], style_feats[3], style_feats[4], style_feats[4]))
        l_identity1 = self.calc_content_loss(Icc, content) + self.calc_content_loss(Iss, style)
        Fcc = self.encode_with_intermediate(Icc)
        Fss = self.encode_with_intermediate(Iss)
        l_identity2 = self.calc_content_loss(Fcc[0], content_feats[0]) + self.calc_content_loss(Fss[0], style_feats[0])
        for i in range(1, 5):
            l_identity2 += self.calc_content_loss(Fcc[i], content_feats[i]) + self.calc_content_loss(Fss[i], style_feats[i])
        

        # Contrastive learning.
        half = int(batch_size / 2)
        style_up = self.style_feature_contrastive(opt1_feats[2][0:half])
        style_down = self.style_feature_contrastive(opt1_feats[2][half:])
        content_up = self.content_feature_contrastive(opt1_feats[3][0:half])
        content_down = self.content_feature_contrastive(opt1_feats[3][half:])

        style_contrastive_loss = 0
        for i in range(half):
            reference_style = style_up[i:i+1]

            if i ==0:
                style_comparisons = torch.cat([style_down[0:half-1], style_up[1:]], 0)
            elif i == 1:
                style_comparisons = torch.cat([style_down[1:], style_up[0:1], style_up[2:]], 0)
            elif i == (half-1):
                style_comparisons = torch.cat([style_down[half-1:], style_down[0:half-2], style_up[0:half-1]], 0)
            else:
                style_comparisons = torch.cat([style_down[i:], style_down[0:i-1], style_up[0:i], style_up[i+1:]], 0)

            style_contrastive_loss += self.compute_contrastive_loss(reference_style, style_comparisons, 0.2, 0)

        for i in range(half):
            reference_style = style_down[i:i+1]

            if i ==0:
                style_comparisons = torch.cat([style_up[0:1], style_up[2:], style_down[1:]], 0)
            elif i == (half-2):
                style_comparisons = torch.cat([style_up[half-2:half-1], style_up[0:half-2], style_down[0:half-2], style_down[half-1:]], 0)
            elif i == (half-1):
                style_comparisons = torch.cat([style_up[half-1:], style_up[1:half-1], style_down[0:half-1]], 0)
            else:
                style_comparisons = torch.cat([style_up[i:i+1], style_up[0:i], style_up[i+2:], style_down[0:i], style_down[i+1:]], 0)

            style_contrastive_loss += self.compute_contrastive_loss(reference_style, style_comparisons, 0.2, 0)


        content_contrastive_loss = 0
        for i in range(half):
            reference_content = content_up[i:i+1]

            if i == 0:
                content_comparisons = torch.cat([content_down[half-1:], content_down[1:half-1], content_up[1:]], 0)
            elif i == 1:
                content_comparisons = torch.cat([content_down[0:1], content_down[2:], content_up[0:1], content_up[2:]], 0)
            elif i == (half-1):
                content_comparisons = torch.cat([content_down[half-2:half-1], content_down[0:half-2], content_up[0:half-1]], 0)
            else:
                content_comparisons = torch.cat([content_down[i-1:i], content_down[0:i-1], content_down[i+1:], content_up[0:i], content_up[i+1:]], 0)

            content_contrastive_loss += self.compute_contrastive_loss(reference_content, content_comparisons, 0.2, 0)

        for i in range(half):
            reference_content = content_down[i:i+1]

            if i == 0:
                content_comparisons = torch.cat([content_up[1:], content_down[1:]], 0)
            elif i == (half-2):
                content_comparisons = torch.cat([content_up[half-1:], content_up[0:half-2], content_down[0:half-2], content_down[half-1:]], 0)
            elif i == (half-1):
                content_comparisons = torch.cat([content_up[0:half-1], content_down[0:half-1]], 0)
            else:
                content_comparisons = torch.cat([content_up[i+1:i+2], content_up[0:i], content_up[i+2:], content_down[0:i], content_down[i+1:]], 0)

            content_contrastive_loss += self.compute_contrastive_loss(reference_content, content_comparisons, 0.2, 0)

        return output1, rc1, rs1, output2, rc2, rs2, l_identity1, l_identity2, content_contrastive_loss, style_contrastive_loss, content_transitive_loss, style_transitive_loss, style_diff_loss, content_restoration_loss, style_restoration_loss, content_loss, style_loss