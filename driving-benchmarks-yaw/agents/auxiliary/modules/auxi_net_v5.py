import torch
import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import copy

__all__ = ['ResNet', 'resnet18_carla', 'resnet34_carla', 'resnet50_carla']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

model_paths = {
    'resnet34': 'resnet34-333f7ec4.pth',
}

'''
in_planes: input channels
out_planes: output channels
'''

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        # AdaptiveAvgPooling2d는 Batch, Channel은 유지.
        # 원하는 output W, H를 입력하면 kernal_size, stride등을 자동으로 설정하여 연산.
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1   = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        self.ca = ChannelAttention(planes)
        self.sa = SpatialAttention()

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = self.ca(out) * out
        out = self.sa(out) * out

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)

        self.ca = ChannelAttention(planes * 4)
        self.sa = SpatialAttention()

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out = self.ca(out) * out
        out = self.sa(out) * out

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        self.inplanes_2 = self.inplanes
        self.inplanes_3 = self.inplanes
        self.inplanes_4 = self.inplanes
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        # self.layer3_2 = self._make_layer_2(block, 256, layers[2], stride=2)
        # self.layer3_3 = self._make_layer_3(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.layer4_2 = self._make_layer_2(block, 512, layers[3], stride=2)
        # self.layer4_3 = self._make_layer_3(block, 512, layers[3], stride=2)
        # self.layer4_4 = self._make_layer_4(block, 512, layers[3], stride=2)

        # self.avgpool = nn.AvgPool2d(7, stride=1)
        # self.fc = nn.Linear(512 * block.expansion, num_classes)

        ''' specialized layers '''
        self.fc_specialize_1 = nn.Sequential(
            nn.Linear(10752, 512),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.Dropout(0.5),
            nn.ReLU(),
        )
        self.fc_specialize_2 = nn.Sequential(
            nn.Linear(10752, 512),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.Dropout(0.5),
            nn.ReLU(),
        )
        # self.fc_specialize_3 = nn.Sequential(
        #     nn.Linear(10752, 512),
        #     nn.Dropout(0.5),
        #     nn.ReLU(),
        #     nn.Linear(512, 512),
        #     nn.Dropout(0.5),
        #     nn.ReLU(),
        # )
        # self.fc_specialize_4 = nn.Sequential(
        #     nn.Linear(10752, 512),
        #     nn.Dropout(0.5),
        #     nn.ReLU(),
        #     nn.Linear(512, 512),
        #     nn.Dropout(0.5),
        #     nn.ReLU(),
        # )

        ''' speed prediction branch '''
        self.speed_branch = nn.Sequential(
            nn.Linear(512, 256),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(256, 256),
            # nn.Dropout(self.dropout_vec[1]),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

        ''' longitudial prediction branch '''
        self.longi_branch = nn.ModuleList([
            nn.Sequential(
                nn.Linear(512, 256),
                nn.Dropout(0.5),
                nn.ReLU(),
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.Linear(256, 2),
            ) for i in range(4)
        ])

        ''' lateral prediction branch '''
        self.later_branch = nn.ModuleList([
            nn.Sequential(
                nn.Linear(512, 256),
                nn.Dropout(0.5),
                nn.ReLU(),
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.Linear(256, 1),
            ) for i in range(4)
        ])

        ''' speed input fc '''
        self.speed_fc = nn.Sequential(
            nn.Linear(1, 128),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.Dropout(0.5),
            nn.ReLU(),
        )

        ''' embedding fc '''
        self.emb_fc_late = nn.Sequential(
            nn.Linear(512 + 128, 512),
            nn.Dropout(0.5),
            nn.ReLU(),
        )

        # ''' posi prediction branch '''
        # self.posi_branch = nn.Sequential(
        #     nn.Linear(512, 256),
        #     nn.Dropout(0.5),
        #     nn.ReLU(),
        #     nn.Linear(256, 256),
        #     # nn.Dropout(self.dropout_vec[1]),
        #     nn.ReLU(),
        #     nn.Linear(256, 2),
        # )
        #
        # ''' yaw prediction branch '''
        # self.yaw_branch = nn.Sequential(
        #     nn.Linear(512, 256),
        #     nn.Dropout(0.5),
        #     nn.ReLU(),
        #     nn.Linear(256, 256),
        #     # nn.Dropout(self.dropout_vec[1]),
        #     nn.ReLU(),
        #     nn.Linear(256, 1),
        # )
        #
        # ''' Relation '''
        # ''' adaptor '''
        # self.adaptor_longi_layer4 = nn.Sequential(
        #     nn.Conv2d(512, 512, kernel_size=1, stride=1, bias=False),
        #     nn.BatchNorm2d(512),
        #     nn.ReLU(),
        # )
        # self.adaptor_late_layer4 = nn.Sequential(
        #     nn.Conv2d(512, 512, kernel_size=1, stride=1, bias=False),
        #     nn.BatchNorm2d(512),
        #     nn.ReLU(),
        # )
        # self.adaptor_posi_layer4 = nn.Sequential(
        #     nn.Conv2d(512, 512, kernel_size=1, stride=1, bias=False),
        #     nn.BatchNorm2d(512),
        #     nn.ReLU(),
        # )
        # self.adaptor_yaw_layer4 = nn.Sequential(
        #     nn.Conv2d(512, 512, kernel_size=1, stride=1, bias=False),
        #     nn.BatchNorm2d(512),
        #     nn.ReLU(),
        # )
        # ''' aggregator '''
        # self.aggregator_layer4_1 = nn.Sequential(
        #     nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.BatchNorm2d(512),
        #     nn.ReLU(),
        # )
        #
        # self.aggregator_layer4_2 = nn.Sequential(
        #     nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.BatchNorm2d(512),
        #     nn.ReLU(),
        # )
        # ''' specialize '''
        # self.fc_specialize_adap_1 = nn.Sequential(
        #     nn.Linear(10752, 512),
        #     nn.Dropout(0.5),
        #     nn.ReLU(),
        #     nn.Linear(512, 512),
        #     nn.Dropout(0.5),
        #     nn.ReLU(),
        # )
        # self.fc_specialize_adap_2 = nn.Sequential(
        #     nn.Linear(10752, 512),
        #     nn.Dropout(0.5),
        #     nn.ReLU(),
        #     nn.Linear(512, 512),
        #     nn.Dropout(0.5),
        #     nn.ReLU(),
        # )
        # ''' rela prediction '''
        # self.rela_later_branch = nn.ModuleList([
        #     nn.Sequential(
        #         nn.Linear(512, 256),
        #         nn.Dropout(0.5),
        #         nn.ReLU(),
        #         nn.Linear(256, 256),
        #         nn.ReLU(),
        #         nn.Linear(256, 1),
        #     ) for i in range(4)
        # ])
        #
        # ''' embedding fc '''
        # self.rela_emb_fc_late = nn.Sequential(
        #     nn.Linear(512 + 128, 512),
        #     nn.Dropout(0.5),
        #     nn.ReLU(),
        # )
        #
        # self.rela_speed_branch = nn.Sequential(
        #     nn.Linear(512, 256),
        #     nn.Dropout(0.5),
        #     nn.ReLU(),
        #     nn.Linear(256, 256),
        #     # nn.Dropout(self.dropout_vec[1]),
        #     nn.ReLU(),
        #     nn.Linear(256, 1),
        # )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(
                    m.weight)
                m.bias.data.fill_(0.01)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes_2 = self.inplanes # 동일한 형태의 layer branch를 만들기 위해 변경 전의 self.inplanes를 저장
        self.inplanes_3 = self.inplanes # 동일한 형태의 layer branch를 만들기 위해 변경 전의 self.inplanes를 저장
        self.inplanes_4 = self.inplanes # 동일한 형태의 layer branch를 만들기 위해 변경 전의 self.inplanes를 저장
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _make_layer_2(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes_2 != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes_2, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes_2, planes, stride, downsample))
        self.inplanes_2 = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes_2, planes))

        return nn.Sequential(*layers)

    def _make_layer_3(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes_3 != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes_3, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes_3, planes, stride, downsample))
        self.inplanes_3 = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes_3, planes))

        return nn.Sequential(*layers)

    def _make_layer_4(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes_4 != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes_4, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes_4, planes, stride, downsample))
        self.inplanes_4 = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes_4, planes))

        return nn.Sequential(*layers)

    def forward(self, x, speed):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        ''' speed input '''
        ipt_speed = self.speed_fc(speed)

        ''' for longitudial '''
        x_longi = self.layer4(x)
        # x_longi_adp = self.adaptor_longi_layer4(x_longi)
        x_longi = x_longi.view(x_longi.size(0), -1)
        x_longi = self.fc_specialize_1(x_longi)
        pred_speed = self.speed_branch(x_longi)
        pred_longi = torch.cat([out(x_longi) for out in self.longi_branch], dim=1)

        ''' for lateral '''
        x_late = self.layer4_2(x)
        # x_late_adp = self.adaptor_late_layer4(x_late)
        x_late = x_late.view(x_late.size(0), -1)
        x_late = self.fc_specialize_2(x_late)
        emb_late = torch.cat([x_late, ipt_speed], dim=1)
        emb_late = self.emb_fc_late(emb_late)
        pred_late = torch.cat([out(emb_late) for out in self.later_branch], dim=1)

        # ''' for position '''
        # x_posi = self.layer4_3(x)
        # x_posi_adp = self.adaptor_posi_layer4(x_posi)
        # x_posi = x_posi.view(x_posi.size(0), -1)
        # x_posi = self.fc_specialize_3(x_posi)
        # pred_posi = self.posi_branch(x_posi)
        #
        # ''' for yaw '''
        # x_yaw = self.layer4_4(x)
        # x_yaw_adp = self.adaptor_yaw_layer4(x_yaw)
        # x_yaw = x_yaw.view(x_yaw.size(0), -1)
        # x_yaw = self.fc_specialize_4(x_yaw)
        # pred_yaw = self.yaw_branch(x_yaw)
        #
        # x_longi_yaw = x_longi_adp + x_yaw_adp
        # x_longi_yaw = self.aggregator_layer4_1(x_longi_yaw)
        # x_longi_yaw = x_longi_yaw.view(x_longi_yaw.size(0), -1)
        # x_longi_yaw = self.fc_specialize_adap_1(x_longi_yaw)
        # emb_longi_yaw = torch.cat([x_longi_yaw, ipt_speed], dim=1)
        # emb_longi_yaw = self.rela_emb_fc_late(emb_longi_yaw)
        # pred_rela_late = torch.cat([out(emb_longi_yaw) for out in self.rela_later_branch], dim=1)
        #
        # x_late_posi = x_posi_adp + x_late_adp
        # x_late_posi = self.aggregator_layer4_2(x_late_posi)
        # x_late_posi = x_late_posi.view(x_late_posi.size(0), -1)
        # x_late_posi = self.fc_specialize_adap_2(x_late_posi)
        # pred_rela_speed = self.rela_speed_branch(x_late_posi)
        #
        # return pred_speed, pred_longi, pred_late, pred_posi, pred_yaw, pred_rela_late, pred_rela_speed
        return pred_speed, pred_longi, pred_late

def resnet18_carla(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        pretrained_state_dict = model_zoo.load_url(model_urls['resnet18'])
        now_state_dict        = model.state_dict()
        now_state_dict.update(pretrained_state_dict)
        model.load_state_dict(now_state_dict)
    return model


def resnet34_carla(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        now_state_dict = model.state_dict()
        # pretrained_state_dict = model_zoo.load_url(model_urls['resnet34'])
        pretrained_state_dict = torch.load(model_paths['resnet34'])
        # model_paths

        # 1. filter out unnecessary keys
        pretrained_state_dict = {k: v for k, v in pretrained_state_dict.items() if k in now_state_dict}
        pretrained_state_dict_layer4_2 = {k.replace("layer4", "layer4_2"): v for k, v in pretrained_state_dict.items()
                                          if k.find("layer4") >= 0 and k.replace("layer4", "layer4_2") in now_state_dict}
        pretrained_state_dict_layer4_3 = {k.replace("layer4", "layer4_3"): v for k, v in pretrained_state_dict.items()
                                          if k.find("layer4") >= 0 and k.replace("layer4", "layer4_3") in now_state_dict}
        pretrained_state_dict_layer4_4 = {k.replace("layer4", "layer4_4"): v for k, v in pretrained_state_dict.items()
                                          if k.find("layer4") >= 0 and k.replace("layer4", "layer4_4") in now_state_dict}
        # 2. overwrite entries in the existing state dict
        now_state_dict.update(pretrained_state_dict)
        now_state_dict.update(pretrained_state_dict_layer4_2)
        now_state_dict.update(pretrained_state_dict_layer4_3)
        now_state_dict.update(pretrained_state_dict_layer4_4)
        # 3. load the new state dict
        model.load_state_dict(now_state_dict)
    return model


def resnet50_carla(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        now_state_dict = model.state_dict()
        # pretrained_state_dict = model_zoo.load_url(model_urls['resnet34'])
        pretrained_state_dict = torch.load(model_paths['resnet34'])
        # model_paths

        # 1. filter out unnecessary keys
        pretrained_state_dict = {k: v for k, v in pretrained_state_dict.items() if k in now_state_dict}
        # 2. overwrite entries in the existing state dict
        now_state_dict.update(pretrained_state_dict)
        # 3. load the new state dict
        model.load_state_dict(now_state_dict)
    return model
