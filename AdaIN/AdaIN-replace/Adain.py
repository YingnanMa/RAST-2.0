import torch.nn as nn
import torch
from function import adaptive_instance_normalization as adain
from function import calc_mean_std

def mean_variance_norm(feat):
    size = feat.size()
    mean, std = calc_mean_std(feat)
    normalized_feat = (feat - mean.expand(size)) / std.expand(size)
    return normalized_feat
    
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

class Net(nn.Module):
    def __init__(self, encoder, decoder):
        super(Net, self).__init__()
        enc_layers = list(encoder.children())
        self.enc_1 = nn.Sequential(*enc_layers[:4])  # input -> relu1_1
        self.enc_2 = nn.Sequential(*enc_layers[4:11])  # relu1_1 -> relu2_1
        self.enc_3 = nn.Sequential(*enc_layers[11:18])  # relu2_1 -> relu3_1
        self.enc_4 = nn.Sequential(*enc_layers[18:31])  # relu3_1 -> relu4_1
        self.decoder = decoder
        self.mse_loss = nn.MSELoss()

        # fix the encoder
        for name in ['enc_1', 'enc_2', 'enc_3', 'enc_4']:
            for param in getattr(self, name).parameters():
                param.requires_grad = False

    # extract relu1_1, relu2_1, relu3_1, relu4_1 from input image
    def encode_with_intermediate(self, input):
        results = [input]
        for i in range(4):
            func = getattr(self, 'enc_{:d}'.format(i + 1))
            results.append(func(results[-1]))
        return results[1:]

    # extract relu4_1 from input image
    def encode(self, input):
        for i in range(4):
            input = getattr(self, 'enc_{:d}'.format(i + 1))(input)
        return input
    '''
    def calc_content_loss(self, input, target):
        assert (input.size() == target.size())
        print(target.requires_grad)
        assert (target.requires_grad is False)
        return self.mse_loss(input, target)

    def calc_style_loss(self, input, target):
        assert (input.size() == target.size())
        assert (target.requires_grad is False)
        input_mean, input_std = calc_mean_std(input)
        target_mean, target_std = calc_mean_std(target)
        return self.mse_loss(input_mean, target_mean) + \
               self.mse_loss(input_std, target_std)
    '''
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

    def forward(self, content, style, alpha=1.0):
        assert 0 <= alpha <= 1
        style_feats = self.encode_with_intermediate(style)
        content_feats = self.encode_with_intermediate(content)

        # restoration right
        stylized1 = adain(content_feats[-1], style_feats[-1])
        stylized1 = alpha * stylized1 + (1 - alpha) * content_feats[-1]
        output1 =  self.decoder(stylized1)
        opt1_feats = self.encode_with_intermediate(output1)

        stylized_rc1 = adain(opt1_feats[-1], content_feats[-1])
        stylized_rc1 = alpha * stylized_rc1 + (1 - alpha) * opt1_feats[-1]
        rc1 = self.decoder(stylized_rc1)
        rc1_feats = self.encode_with_intermediate(rc1)

        stylized_rs1 = adain(style_feats[-1], opt1_feats[-1])
        stylized_rs1 = alpha * stylized_rs1 + (1 - alpha) * style_feats[-1]
        rs1 = self.decoder(stylized_rs1)
        rs1_feats = self.encode_with_intermediate(rs1)

        # restoration left
        stylized2 = adain(style_feats[-1], content_feats[-1])
        stylized2 = alpha * stylized2 + (1 - alpha) * style_feats[-1]
        output2 = self.decoder(stylized2)
        opt2_feats = self.encode_with_intermediate(output2)

        stylized2_rc2 = adain(content_feats[-1], opt2_feats[-1])
        stylized2_rc2 = alpha * stylized2_rc2 + (1 - alpha) * content_feats[-1]
        rc2 = self.decoder(stylized2_rc2)
        rc2_feats = self.encode_with_intermediate(rc2)

        stylized2_rs2 = adain(opt2_feats[-1], style_feats[-1])
        stylized2_rs2 = alpha * stylized2_rs2 + (1 - alpha) * opt2_feats[-1]
        rs2 = self.decoder(stylized2_rs2)
        rs2_feats = self.encode_with_intermediate(rs2)

        # restoration loss functions right
        content_transitive_loss1 = self.calc_content_loss(rc1_feats[-1], content_feats[-1], norm=True)
        style_diff_loss1 = self.calc_style_loss(opt1_feats[0], content_feats[0])
        for i in range(1,4):
            style_diff_loss1 += self.calc_style_loss(opt1_feats[i], content_feats[i])
        style_diff_loss1=1/style_diff_loss1

        style_transitive_loss1 = self.calc_style_loss(rs1_feats[0], style_feats[0])
        for i in range(1,4):
            style_transitive_loss1 += self.calc_style_loss(rs1_feats[i], style_feats[i])
        
        #restoration loss functions left
        content_transitive_loss2 = self.calc_content_loss(rs2_feats[-1], style_feats[-1], norm=True)
        style_diff_loss2 = self.calc_style_loss(opt2_feats[0], style_feats[0])
        for i in range(1,4):
            style_diff_loss2 += self.calc_style_loss(opt2_feats[i], style_feats[i])
        style_diff_loss2 = 1/style_diff_loss2

        style_transitive_loss2 = self.calc_style_loss(rc2_feats[0], content_feats[0])
        for i in range(1,4):
            style_transitive_loss2 += self.calc_style_loss(rc2_feats[i], content_feats[i])

        #restoration loss
        content_restoration_loss = self.calc_content_loss(rc1_feats[-1], rc2_feats[-1], norm=True) + self.calc_content_loss(rs1_feats[-1], rs2_feats[-1], norm=True)
        style_restoration_loss = self.calc_style_loss(rc1_feats[0], rc2_feats[0])
        for i in range(1, 4):
            style_restoration_loss += self.calc_style_loss(rc1_feats[i], rc2_feats[i])
            
        style_restoration_loss += self.calc_style_loss(rs1_feats[0], rs2_feats[0])
        for j in range(1, 4):
            style_restoration_loss += self.calc_style_loss(rs1_feats[j], rs2_feats[j])

        content_transitive_loss = content_transitive_loss1 + content_transitive_loss2
        style_transitive_loss = style_transitive_loss1 + style_transitive_loss2
        style_diff_loss = style_diff_loss1 + style_diff_loss2


        # 问题：
        # 1. 更改思路是否正确
        # 2. encode or encode_with_intermediate 
        # 3. loss: t or content_feats 用哪个
        # 4. identity loss? and contrastive loss? 如何替换
        # 5. adversial loss? 在哪

        return output1, rc1, rs1, output2, rc2, rs2, content_transitive_loss, style_transitive_loss, style_diff_loss, content_restoration_loss, style_restoration_loss

        '''
        style_feats = self.encode_with_intermediate(style)
        content_feats = self.encode(content)
        t = adain(content_feats, style_feats[-1])
        t = alpha * t + (1 - alpha) * content_feats

        g_t = self.decoder(t)
        g_t_feats = self.encode_with_intermediate(g_t)

        loss_c = self.calc_content_loss(g_t_feats[-1], t)
        loss_s = self.calc_style_loss(g_t_feats[0], style_feats[0])
        for i in range(1, 4):
            loss_s += self.calc_style_loss(g_t_feats[i], style_feats[i])
        return loss_c, loss_s
        '''



