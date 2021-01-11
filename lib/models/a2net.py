import torch.nn as nn
import torch
import torch.nn.functional as F


class BaseFeatureNet(nn.Module):
    '''
    calculate feature
    input: [batch_size, 128, 1024]
    output: [batch_size, 32, 512]
    '''
    def __init__(self, cfg):
        super(BaseFeatureNet, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=cfg.MODEL.IN_FEAT_DIM,
                               out_channels=cfg.MODEL.BASE_FEAT_DIM,
                               kernel_size=9, stride=1, padding=4, bias=True)
        self.conv2 = nn.Conv1d(in_channels=cfg.MODEL.BASE_FEAT_DIM,
                               out_channels=cfg.MODEL.BASE_FEAT_DIM,
                               kernel_size=9, stride=1, padding=4, bias=True)
        self.max_pooling = nn.MaxPool1d(kernel_size=2, stride=2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        feat = self.relu(self.conv1(x))
        feat = self.max_pooling(feat)
        feat = self.relu(self.conv2(feat))
        feat = self.max_pooling(feat)
        return feat


class FeatNet(nn.Module):
    '''
    main network
    input: base feature, [batch_size, 32, 512]
    output: MAL1, MAL2, MAL3
    '''
    def __init__(self, cfg):
        super(FeatNet, self).__init__()
        self.base_feature_net = BaseFeatureNet(cfg)
        self.convs = nn.ModuleList()
        for layer in range(cfg.MODEL.NUM_LAYERS):
            # stride = 1 if layer == 0 else 2
            in_channel = cfg.MODEL.BASE_FEAT_DIM if layer == 0 else cfg.MODEL.LAYER_DIMS[layer - 1]
            out_channel = cfg.MODEL.LAYER_DIMS[layer]
            conv = nn.Conv1d(in_channel, out_channel, kernel_size=3, stride=cfg.MODEL.LAYER_STRIDES[layer], padding=1)
            self.convs.append(conv)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        results = []
        feat = self.base_feature_net(x)
        for conv in self.convs:
            feat = self.relu(conv(feat))
            results.append(feat)

        return tuple(results)


class LocNetAB(nn.Module):
    '''
    Action multi-class classification and regression
    input:
    output: class score + conf + location (center & width)
    '''
    def __init__(self, cfg):
        super(LocNetAB, self).__init__()
        self.num_class = cfg.DATASET.NUM_CLASSES
        self.num_pred_value = cfg.DATASET.NUM_CLASSES + 2
        self.num_box = len(cfg.MODEL.ASPECT_RATIOS)
        self.head_stack_layers = cfg.MODEL.HEAD_LAYERS

        self._init_head(cfg)

    def _init_head(self, cfg):
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        for layer in range(self.head_stack_layers):
            in_channel = cfg.MODEL.LAYER_DIMS[-1] if layer == 0 else cfg.MODEL.HEAD_DIM
            out_channel = cfg.MODEL.HEAD_DIM
            cls_conv = nn.Conv1d(in_channel, out_channel, kernel_size=3, padding=1)
            reg_conv = nn.Conv1d(in_channel, out_channel, kernel_size=3, padding=1)
            self.cls_convs.append(cls_conv)
            self.reg_convs.append(reg_conv)
        self.pred_cls = nn.Conv1d(cfg.MODEL.HEAD_DIM, self.num_box * self.num_class, kernel_size=3, padding=1)
        self.pred_reg = nn.Conv1d(cfg.MODEL.HEAD_DIM, self.num_box * 2, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def tensor_view(self, cls, reg):
        '''
        view the tensor for [batch, 120, depth] to [batch, (depth*5), 24]
        make the prediction (24 values) for each anchor box at the last dimension
        '''
        bs, c, t = cls.size()
        cls = cls.view(bs, -1, self.num_class, t).permute(0, 3, 1, 2).contiguous()
        reg = reg.view(bs, -1, 2, t).permute(0, 3, 1, 2).contiguous()
        data = torch.cat((cls, reg), dim=-1)
        data = data.view(bs, -1, self.num_pred_value)
        return data

    def forward_single(self, x):
        cls_feat = x
        reg_feat = x
        for cls_conv in self.cls_convs:
            cls_feat = self.relu(cls_conv(cls_feat))
        for reg_conv in self.reg_convs:
            reg_feat = self.relu(reg_conv(reg_feat))
        pred_cls = self.pred_cls(cls_feat)
        pred_reg = self.pred_reg(reg_feat)
        return self.tensor_view(pred_cls, pred_reg)

    def forward(self, feats):
        return tuple(map(self.forward_single, feats))


############### anchor-free ##############
class PredHeadBranch(nn.Module):
    def __init__(self, cfg):
        super(PredHeadBranch, self).__init__()
        self.head_stack_layers = cfg.MODEL.HEAD_LAYERS
        self._init_head(cfg)

    def _init_head(self, cfg):
        self.convs = nn.ModuleList()
        for layer in range(self.head_stack_layers):
            in_channel = cfg.MODEL.REDU_CHA_DIM if layer == 0 else cfg.MODEL.HEAD_DIM
            out_channel = cfg.MODEL.HEAD_DIM
            conv = nn.Conv1d(in_channel, out_channel, kernel_size=3, padding=1)
            self.convs.append(conv)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        feat = x
        for conv in self.convs:
            feat = self.relu(conv(feat))
        return feat


class PredHead(nn.Module):
    '''
    Predict classification and regression
    '''
    def __init__(self, cfg):
        super(PredHead, self).__init__()
        self.head_branches = nn.ModuleList()
        for _ in range(4):
            self.head_branches.append(PredHeadBranch(cfg))

        num_class = cfg.DATASET.NUM_CLASSES
        num_box = len(cfg.MODEL.ASPECT_RATIOS)

        af_cls = nn.Conv1d(cfg.MODEL.HEAD_DIM, num_class, kernel_size=3, padding=1)
        af_reg = nn.Conv1d(cfg.MODEL.HEAD_DIM, 2, kernel_size=3, padding=1)
        ab_cls = nn.Conv1d(cfg.MODEL.HEAD_DIM, num_box * num_class, kernel_size=3, padding=1)
        ab_reg = nn.Conv1d(cfg.MODEL.HEAD_DIM, num_box * 2, kernel_size=3, padding=1)
        self.pred_heads = nn.ModuleList([af_cls, af_reg, ab_cls, ab_reg])

    def forward(self, x):
        preds = []
        for pred_branch, pred_head in zip(self.head_branches, self.pred_heads):
            feat = pred_branch(x)
            preds.append(pred_head(feat))

        return tuple(preds)


class ReduceChannel(nn.Module):
    def __init__(self, cfg):
        super(ReduceChannel, self).__init__()
        self.convs = nn.ModuleList()
        for layer in range(cfg.MODEL.NUM_LAYERS):
            conv = nn.Conv1d(cfg.MODEL.LAYER_DIMS[layer], cfg.MODEL.REDU_CHA_DIM, kernel_size=1)
            self.convs.append(conv)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, feat_list):
        assert len(feat_list) == len(self.convs)
        results = []
        for conv, feat in zip(self.convs, feat_list):
            results.append(self.relu(conv(feat)))

        return tuple(results)


class LocNet(nn.Module):
    '''
    Predict action boundary, based on features from different FPN levels
    '''
    def __init__(self, cfg):
        super(LocNet, self).__init__()
        # self.features = FeatNet(cfg)
        self.reduce_channels = ReduceChannel(cfg)
        self.pred = PredHead(cfg)
        self.num_class = cfg.DATASET.NUM_CLASSES
        self.ab_pred_value = cfg.DATASET.NUM_CLASSES + 2

    def _layer_cal(self, feat_list):
        af_cls = list()
        af_reg = list()
        ab_pred = list()

        for feat in feat_list:
            cls_af, reg_af, cls_ab, reg_ab = self.pred(feat)
            af_cls.append(cls_af.permute(0, 2, 1).contiguous())
            af_reg.append(reg_af.permute(0, 2, 1).contiguous())
            ab_pred.append(self.tensor_view(cls_ab, reg_ab))

        af_cls = torch.cat(af_cls, dim=1)  # bs, sum(t_i), n_class+1
        af_reg = torch.cat(af_reg, dim=1)  # bs, sum(t_i), 2
        af_reg = F.relu(af_reg)

        return (af_cls, af_reg), tuple(ab_pred)

    def tensor_view(self, cls, reg):
        '''
        view the tensor for [batch, 120, depth] to [batch, (depth*5), 24]
        make the prediction (24 values) for each anchor box at the last dimension
        '''
        bs, c, t = cls.size()
        cls = cls.view(bs, -1, self.num_class, t).permute(0, 3, 1, 2).contiguous()
        reg = reg.view(bs, -1, 2, t).permute(0, 3, 1, 2).contiguous()
        data = torch.cat((cls, reg), dim=-1)
        data = data.view(bs, -1, self.ab_pred_value)
        return data

    def forward(self, features_list):
        features_list = self.reduce_channels(features_list)
        return self._layer_cal(features_list)


class A2Net(nn.Module):
    def __init__(self, cfg):
        super(A2Net, self).__init__()
        self.features = FeatNet(cfg)
        # self.af = LocNetAF(cfg)
        # self.ab = LocNetAB(cfg)
        self.loc_net = LocNet(cfg)

    def forward(self, x):
        features = self.features(x)
        # out_af = self.af(features)
        # out_ab = self.ab(features)
        out_af, out_ab = self.loc_net(features)
        return out_af, out_ab


if __name__ == '__main__':
    import sys
    sys.path.append('/disk/yangle/TIP2020-A2Net/A2Net/lib')
    from config import cfg, update_config
    cfg_file = '/disk/yangle/TIP2020-A2Net/A2Net/experiments/thumos/A2Net.yaml'
    update_config(cfg_file)

    model = LocNet(cfg).cuda()
    data = torch.randn((1, 1024, 128)).cuda()
    output = model(data)
