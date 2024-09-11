import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
sys.path.append('.')

from util.init_func import init_weight
from config import config



class EncoderDecoder(nn.Module):
    def __init__(self,cfg=None, encoder_name='mit_b2', decoder_name='MLPDecoder', n_class=config.num_classes, norm_layer=nn.BatchNorm2d):
        super(EncoderDecoder, self).__init__()
        self.channels = [64, 128, 320, 512]
        self.norm_layer = norm_layer

        # import backbone and decoder
        if encoder_name == 'mit_b5':
            #logger.info('Using backbone: Segformer-B5')
            from model.segformer.encoders.dual_segformer import mit_b5 as backbone
            print("chose mit_b5")
            self.backbone = backbone(norm_fuse=norm_layer)
        elif encoder_name == 'mit_b4':
            #logger.info('Using backbone: Segformer-B4')
            from model.segformer.encoders.dual_segformer import mit_b4 as backbone
            print("chose mit_b4")
            self.backbone = backbone(norm_fuse=norm_layer)
        elif encoder_name == 'mit_b3':
            #logger.info('Using backbone: Segformer-B4')
            from model.segformer.encoders.dual_segformer import mit_b3 as backbone
            print("chose mit_b3")
            self.backbone = backbone(norm_fuse=norm_layer)
        elif encoder_name == 'mit_b2':
            #logger.info('Using backbone: Segformer-B2')
            from model.segformer.encoders.dual_segformer import mit_b2 as backbone
            print("chose mit_b2")
            self.backbone = backbone(norm_fuse=norm_layer)
        elif encoder_name == 'mit_b1':
            #logger.info('Using backbone: Segformer-B1')
            from model.segformer.encoders.dual_segformer import mit_b1 as backbone
            print("chose mit_b1")
            self.backbone = backbone(norm_fuse=norm_layer)
        elif encoder_name == 'mit_b0':
            #logger.info('Using backbone: Segformer-B0')
            self.channels = [32, 64, 160, 256]
            from model.segformer.encoders.dual_segformer import mit_b0 as backbone
            print("chose mit_b0")
            self.backbone = backbone(norm_fuse=norm_layer)
        else:
            #logger.info('Using backbone: Segformer-B2')
            from encoders.dual_segformer import mit_b2 as backbone
            self.backbone = backbone(norm_fuse=norm_layer)

        self.aux_head = None

        if decoder_name == 'MLPDecoder':
            #logger.info('Using MLP Decoder')
            from model.segformer.decoders.MLPDecoder import DecoderHead
            self.decode_head = DecoderHead(in_channels=self.channels, num_classes=n_class, norm_layer=norm_layer, embed_dim=512)
        elif decoder_name == 'MLPDecoderaddition':
            #logger.info('Using MLP Decoder')
            from model.segformer.decoders.MLPDecoderaddition import DecoderHead
            self.decode_head = DecoderHead(in_channels=self.channels, num_classes=n_class, norm_layer=norm_layer, embed_dim=512)
        else:
            #logger.info('No decoder(FCN-32s)')
            from decoders.fcnhead import FCNHead
            self.decode_head = FCNHead(in_channels=self.channels[-1], kernel_size=3, num_classes=10, norm_layer=norm_layer)

        self.voting = nn.Conv2d(in_channels=n_class*2,out_channels=n_class,kernel_size=3,stride=1,padding=1)

        self.init_weights(cfg, pretrained=cfg.pretrained_model)
    
    def init_weights(self, cfg, pretrained=None):
        if pretrained:
            self.backbone.init_weights(pretrained=pretrained)
        init_weight(self.decode_head, nn.init.kaiming_normal_,
                self.norm_layer, cfg.bn_eps, cfg.bn_momentum,
                mode='fan_in', nonlinearity='relu')

    def encode_decode(self, rgb, modal_x):
        """Encode images with backbone and decode into a semantic segmentation
        map of the same size as input."""
        orisize = rgb.shape
        x = self.backbone(rgb, modal_x)
        out = self.decode_head.forward(x)
        out = F.interpolate(out, size=orisize[2:], mode='bilinear', align_corners=False)
        return out

    def forward(self, input):

        rgb = input[:,:3]
        modal_x = input[:,3:]
        modal_x = torch.cat((modal_x, modal_x, modal_x),dim=1)

        out = self.encode_decode(rgb, modal_x)
        return out
