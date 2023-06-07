'''
使用paddle实现的U-Net网络
'''

import paddle 
import paddle.nn as nn


class DoubleConv(nn.Layer):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2D(in_channels, out_channels, kernel_size=3, padding=1, bias_attr=True),
            nn.BatchNorm2D(out_channels),
            nn.ReLU(),
            nn.Conv2D(out_channels, out_channels, kernel_size=3, padding=1, bias_attr=True),
            nn.ReLU()
        )
    
    def forward(self, x):
        x = self.conv(x)
        return x


class UNet(nn.Layer):
    def __init__(self, in_channels=3, out_channels=2):
        super(UNet, self).__init__()

        out_ch = [64, 128, 256, 512, 1024]

        # 编码器，下采样
        self.encode_conv1 = DoubleConv(in_channels, out_ch[0])
        self.encode_conv2 = DoubleConv(out_ch[0], out_ch[1])
        self.encode_conv3 = DoubleConv(out_ch[1], out_ch[2])
        self.encode_conv4 = DoubleConv(out_ch[2], out_ch[3])
        self.encode_conv5 = DoubleConv(out_ch[3], out_ch[4])

        self.pool1 = nn.MaxPool2D(kernel_size=2, stride=2)
        self.pool2 = nn.MaxPool2D(kernel_size=2, stride=2)
        self.pool3 = nn.MaxPool2D(kernel_size=2, stride=2)
        self.pool4 = nn.MaxPool2D(kernel_size=2, stride=2)

        # 解码器，用逆卷积上采样
        self.decode_conv1 = DoubleConv(out_ch[4], out_ch[3])
        self.decode_conv2 = DoubleConv(out_ch[3], out_ch[2])
        self.decode_conv3 = DoubleConv(out_ch[2], out_ch[1])
        self.decode_conv4 = DoubleConv(out_ch[1], out_ch[0])

        self.out = nn.Conv2D(out_ch[0], out_channels, kernel_size=1)

        self.up1 = nn.Conv2DTranspose(out_ch[4], out_ch[3], kernel_size=2, stride=2)
        self.up2 = nn.Conv2DTranspose(out_ch[3], out_ch[2], kernel_size=2, stride=2)
        self.up3 = nn.Conv2DTranspose(out_ch[2], out_ch[1], kernel_size=2, stride=2)
        self.up4 = nn.Conv2DTranspose(out_ch[1], out_ch[0], kernel_size=2, stride=2)

    def forward(self, x):
        # 编码器
        x1 = self.encode_conv1(x)
        x2 = self.encode_conv2(self.pool1(x1))
        x3 = self.encode_conv3(self.pool2(x2))
        x4 = self.encode_conv4(self.pool3(x3))
        x5 = self.encode_conv5(self.pool4(x4))

        # 解码器
        x6 = self.up1(x5)
        x6 = paddle.concat([x6, x4], axis=1)
        x6 = self.decode_conv1(x6)
        x7 = self.up2(x6)
        x7 = paddle.concat([x7, x3], axis=1)
        x7 = self.decode_conv2(x7)
        x8 = self.up3(x7)
        x8 = paddle.concat([x8, x2], axis=1)
        x8 = self.decode_conv3(x8)
        x9 = self.up4(x8)
        x9 = paddle.concat([x9, x1], axis=1)
        x9 = self.decode_conv4(x9)

        x = self.out(x9)
        return x
