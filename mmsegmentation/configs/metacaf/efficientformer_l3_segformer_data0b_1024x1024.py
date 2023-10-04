# Commonly Modified Hyperparameters

num_classes = 2
crop_size = (1024, 1024)
train_batch_size = 6
train_num_workers = 4
max_iters=80000
val_interval=500
dataset_type = 'ArtifactsDataset'
data_root = '/sensei-fs/users/lingzzha/projects/AdobePAL/data/metacaf/adobepal_datasets/data0b'
train_img_path = 'images/train'
train_seg_map_path='labels/train'
val_img_path = 'images/val'
val_seg_map_path='labels/val'
test_img_path = 'images/test'
test_seg_map_path='labels/test'
bonebone_type = 'efficientformer_l3_feat'
head_type = 'SegformerHead_x2'



""" Model """

# model settings
resolution = crop_size[0] # assume square image
norm_cfg = dict(type='SyncBN', requires_grad=True)

data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255,
    size=crop_size)

model = dict(
    type='EncoderDecoder',
    data_preprocessor=data_preprocessor,

    pretrained=None,
    backbone=dict(
        type=bonebone_type,
        style='pytorch',
        resolution = resolution,
        init_cfg=dict(
            type='Pretrained',
            checkpoint='/sensei-fs/tenants/Sensei-DITeam/zwei-share/share-with-lingzhi/efficientformer_l3_300d_new.pth',
        )
        ),
    
    # default one; Zijun has Segformer_2x
    decode_head=dict(
        type=head_type, # x4 -> x2
        in_channels=[64, 128, 320, 512],
        in_index=[0, 1, 2, 3],
        channels=256,
        dropout_ratio=0.1,
        num_classes=num_classes,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)
        # loss_decode=[dict(type='CrossEntropyLoss', use_sigmoid=False, loss_name='loss_ce', loss_weight=1.0),
        #     dict(type='DiceLoss', loss_name='loss_dice', loss_weight=3.0)]
        ),
    
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))


""" Dataset """
# dataset settings
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(
        type='RandomResize',
        scale=crop_size,
        ratio_range=(0.5, 2.0),
        keep_ratio=True),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    # dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs')
]
test_pipeline = [ 
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=crop_size, keep_ratio=True),
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs')
]
train_dataloader = dict(
    batch_size=train_batch_size,
    num_workers=train_num_workers,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path=train_img_path, seg_map_path=train_seg_map_path),
        pipeline=train_pipeline))
val_dataloader = dict(
    batch_size=6,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path=val_img_path, seg_map_path=val_seg_map_path),
        pipeline=test_pipeline))
test_dataloader = dict(
    batch_size=6,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path=test_img_path, seg_map_path=test_seg_map_path),
        pipeline=test_pipeline))

val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'])
test_evaluator = val_evaluator



""" Optimization (Learning Rate, Iterations, etc.) """

# optimizer
optim_wrapper = dict(
    # _delete_=True,
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW', lr=0.00006, betas=(0.9, 0.999), weight_decay=0.01),
    paramwise_cfg=dict(
        custom_keys={
            'pos_block': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.),
            'head': dict(lr_mult=10.)
        }))

param_scheduler = [
    dict(
        type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=1500),
    dict(
        type='PolyLR',
        eta_min=0.0,
        power=1.0,
        begin=1500,
        end=160000,
        by_epoch=False,
    )
]


""" Runtime """
default_scope = 'mmseg'
env_cfg = dict(
    cudnn_benchmark=True,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)
vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='SegLocalVisualizer', vis_backends=vis_backends, name='visualizer')
log_processor = dict(by_epoch=False)
log_level = 'INFO'
load_from = None
resume = False


""" Training Configurations """
# training schedule for 40k
train_cfg = dict(type='IterBasedTrainLoop', max_iters=max_iters, val_interval=val_interval)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', by_epoch=False, save_best='mIoU'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook'))


""" Evaluation and Checkpoint """
val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU']) # , save_best='mIoU'
checkpoint_config = dict(by_epoch=False)



# loss: aux head? dice? 
# data aug 
# torchscript convert 
# pretrain? 