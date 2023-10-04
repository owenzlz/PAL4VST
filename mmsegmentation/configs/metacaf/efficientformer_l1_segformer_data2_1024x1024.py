_base_ = [
    'efficientformer_l3_segformer_data2_1024x1024.py'
]



model = dict(

    pretrained=None,
    backbone=dict(
        type='efficientformer_l1_feat',
        style='pytorch',
        # init_cfg=dict(
        #     type='Pretrained',
        #     checkpoint='/sensei-fs/tenants/Sensei-DITeam/zwei-share/share-with-lingzhi/efficientformer_l3_300d_new.pth',
        # )
        ),
    
    # default one; Zijun has Segformer_2x
    decode_head=dict(
        type='SegformerHead_x2', 
        in_channels=[48, 96, 224, 448],
        in_index=[0, 1, 2, 3],
        channels=256,
        dropout_ratio=0.1,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)
        # loss_decode=[dict(type='CrossEntropyLoss', use_sigmoid=False, loss_name='loss_ce', loss_weight=1.0),
        #     dict(type='DiceLoss', loss_name='loss_dice', loss_weight=3.0)]
        ),
    
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))




