import torch.nn as nn

cfg = {
    'vgg9': [128, 128, 'M', 256, 256, 'M', 512, 512, 'M'],
    'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

class VGG(nn.Module):
    def __init__(self, vgg_name, num_classes=10):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)

#foodsource: 1d向量，值为每层通道数
class PSOVGG(nn.Module):
    def __init__(self, vgg_name, foodsource):
        super(PSOVGG, self).__init__()
        self.foodsource = foodsource
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear((foodsource[len(foodsource)-1]), 10)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out
        
    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        #index = 0
        Mlayers = 0
        for x_index, x in enumerate(cfg):
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
                Mlayers += 1
            else:
                x = self.foodsource[x_index - Mlayers]
                if x == 0:
                    x = 1
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)


'''
#foodsource: 1d向量，值限制在0-9
class BeeVGG(nn.Module):
    def __init__(self, vgg_name, foodsource):
        super(BeeVGG, self).__init__()
        self.foodsource = foodsource
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(int(512 * foodsource[len(foodsource)-1] / 10), 10)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out
        
    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        #index = 0
        Mlayers = 0
        for x_index, x in enumerate(cfg):
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
                Mlayers += 1
            else:
                x = int(x * self.foodsource[x_index - Mlayers] / 10)
                if x == 0:
                    x = 1
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)
'''
