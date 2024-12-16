model = dict(
    name='MADNet',
    type='EncoderDecoder',
    pretrained=None,
    backbone=dict(
        type=None
    ),
    decode_head=dict(
        type='MADNet',
        in_ch=3,
        out_ch=1,
        dim=64,
        ori_h=256,
        deep_supervision=True
    ),
    loss=dict(type='SoftIoULoss')
)
