_base_ = [
    '../../_base_/datasets/mmdet/coco_detection.py',
    '../../_base_/schedules/mmdet/schedule_2x.py',
    '../../_base_/mmdet_runtime.py'
]

custom_imports = dict(imports=['SDAKD'], allow_failed_imports=False)


# model settings
student = dict(
    type='mmdet.FCOS',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        style='caffe',
        init_cfg=dict(
            type='Pretrained',
            checkpoint='open-mmlab://detectron/resnet50_caffe')),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_output',  # use P5
        num_outs=5,
        relu_before_extra_convs=True),
    bbox_head=dict(
        type='FCOSHead',
        num_classes=80,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        strides=[8, 16, 32, 64, 128],
        # add changes following previous KD methods
        norm_on_bbox=True,
        centerness_on_reg=False,
        dcn_on_last_conv=False,
        center_sampling=True,
        conv_bias=True,
        loss_bbox=dict(type='OSDGIoULoss', loss_weight=1.0),
        # end change
        loss_cls=dict(
            type='OSDFocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        # loss_bbox=dict(type='IoULoss', loss_weight=1.0),
        loss_centerness=dict(
            type='OSDCrossEntropyLoss', use_sigmoid=True, loss_weight=1.0)),
    # training and testing settings
    train_cfg=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.5,
            neg_iou_thr=0.4,
            min_pos_iou=0,
            ignore_iof_thr=-1),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        # nms=dict(type='nms', iou_threshold=0.5),
        # add change following previous KD methods
        nms=dict(type='nms', iou_threshold=0.6),
        max_per_img=100))

teacher = dict(
    type='mmdet.FCOS',
    init_cfg=dict(type='Pretrained', checkpoint='https://download.openmmlab.com/mmdetection/v2.0/fcos/fcos_r101_caffe_fpn_gn-head_mstrain_640-800_2x_coco/fcos_r101_caffe_fpn_gn-head_mstrain_640-800_2x_coco-511424d6.pth'),
    backbone=dict(
        type='ResNet',
        depth=101,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        style='caffe',),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_output',  # use P5
        num_outs=5,
        relu_before_extra_convs=True),
    bbox_head=dict(
        type='FCOSHead',
        num_classes=80,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        strides=[8, 16, 32, 64, 128],
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='IoULoss', loss_weight=1.0),
        loss_centerness=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0)),
    # training and testing settings
    train_cfg=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.5,
            neg_iou_thr=0.4,
            min_pos_iou=0,
            ignore_iof_thr=-1),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.5),
        max_per_img=100))

algorithm = dict(
    type='OSDDistill',
    solve_number=4,
    convertor_training_epoch=[6,12],
    convertor_epoch_number=1,
    pretrain_path='/home/sst/product/SDAKD/SDAKD_FOR_BASELINE/checkpoints/Augmentation',
    collect_key=['img', 'gt_bboxes', 'gt_labels'],
    architecture=dict(
        type='MMDetArchitecture',
        model=student,
    ),
    distiller=dict(
        type='OSDTeacherDistiller',
        teacher=teacher,
        teacher_trainable=False,
        components=[
            dict(
                student_module='bbox_head.conv_cls',
                teacher_module='bbox_head.conv_cls',
                losses=[
                    dict(
                        type='KLDivergence',
                        name='loss_kd_conv_cls_head',
                        tau=1,
                        loss_weight=1.,
                    )
                ]),
            dict(
                student_module='bbox_head.conv_centerness',
                teacher_module='bbox_head.conv_centerness',
                losses=[
                    dict(
                        type='KLDivergence',
                        name='loss_kd_conv_centerness_head',
                        tau=1,
                        loss_weight=1.,
                    )
                ]),
            dict(
                student_module='neck',
                teacher_module='neck',
                losses=[
                    dict(
                        type='MSELoss',
                        name='loss_mse_fpn',
                        loss_weight=1,
                    )
                ],
                align_module=dict(
                    type='conv2d',
                    num_modules=5,
                    student_channels=256,
                    teacher_channels=256,
                )
            ),
        ]),

)

find_unused_parameters = True

img_norm_cfg = dict(
    mean=[102.9801, 115.9465, 122.7717], std=[1.0, 1.0, 1.0], to_rgb=False)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
        dict(
        type='Resize',
        img_scale=[(1333, 640), (1333, 800)],
        multiscale_mode='value',
        keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(pipeline=train_pipeline),
    val=dict(pipeline=test_pipeline),
    test=dict(pipeline=test_pipeline))

# optimizer

optimizer = dict(lr=0.01)
optimizer_config = dict(_delete_=True, grad_clip=dict(max_norm=35, norm_type=2))

# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    step=[16, 22]
)

runner = dict(type='OSDBasedRunner', max_epochs=24)
fp16 = dict(loss_scale=512.)

