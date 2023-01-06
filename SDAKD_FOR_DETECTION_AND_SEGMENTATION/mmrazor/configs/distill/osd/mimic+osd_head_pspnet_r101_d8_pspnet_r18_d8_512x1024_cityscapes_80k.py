_base_ = [
    '../../_base_/datasets/mmseg/cityscapes.py',
    '../../_base_/mmseg_runtime.py',
    '../../_base_/schedules/mmseg/schedule_80k.py'
]

custom_imports = dict(imports=['SDAKD'], allow_failed_imports=False)

norm_cfg = dict(type='SyncBN', requires_grad=True)

# pspnet r18
student = dict(
    type='mmseg.EncoderDecoder',
    backbone=dict(
        type='ResNetV1c',
        init_cfg=dict(
            type='Pretrained', checkpoint='open-mmlab://resnet18_v1c'),
        depth=18,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        dilations=(1, 1, 2, 4),
        strides=(1, 2, 1, 1),
        norm_cfg=norm_cfg,
        norm_eval=False,
        style='pytorch',
        contract_dilation=True),
    decode_head=dict(
        type='PSPHead',
        in_channels=512,
        in_index=3,
        channels=128,
        pool_scales=(1, 2, 3, 6),
        dropout_ratio=0.1,
        num_classes=19,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='OSDSEGCrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=256,
        in_index=2,
        channels=64,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=19,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='OSDSEGCrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))

checkpoint = 'https://download.openmmlab.com/mmsegmentation/v0.5/pspnet/pspnet_r101-d8_512x1024_80k_cityscapes/pspnet_r101-d8_512x1024_80k_cityscapes_20200606_112211-e1e1100f.pth'  # noqa: E501

# pspnet r101
teacher = dict(
    type='mmseg.EncoderDecoder',
    init_cfg=dict(type='Pretrained', checkpoint=checkpoint),
    backbone=dict(
        type='ResNetV1c',
        depth=101,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        dilations=(1, 1, 2, 4),
        strides=(1, 2, 1, 1),
        norm_cfg=norm_cfg,
        norm_eval=False,
        style='pytorch',
        contract_dilation=True),
    decode_head=dict(
        type='PSPHead',
        in_channels=2048,
        in_index=3,
        channels=512,
        pool_scales=(1, 2, 3, 6),
        dropout_ratio=0.1,
        num_classes=19,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=1024,
        in_index=2,
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=19,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),

)

# algorithm = dict(
#     type='GeneralDistill',
#     architecture=dict(
#         type='MMSegArchitecture',
#         model=student,
#     ),
#     distiller=dict(
#         type='SingleTeacherDistiller',
#         teacher=teacher,
#         teacher_trainable=False,
#         components=[
#             dict(
#                 student_module='decode_head.conv_seg',
#                 teacher_module='decode_head.conv_seg',
#                 losses=[
#                     dict(
#                         type='ChannelWiseDivergence',
#                         name='loss_cwd_logits',
#                         tau=1,
#                         loss_weight=5,
#                     )
#                 ])
#         ]),
# )

runner = dict(type='OSDIterBasedRunner', max_iters=80000)
checkpoint_config = dict(by_epoch=False, interval=1000)
find_unused_parameters = True

algorithm = dict(
    type='OSDDistill',
    solve_number=4,
    convertor_training_epoch=[0,6,12],
    convertor_epoch_number=1,
    pretrain_path='/home/sst/product/SDAKD/SDAKD_FOR_BASELINE/checkpoints/Augmentation',
    collect_key=['img', 'gt_bboxes', 'gt_labels', "gt_semantic_seg"],
    architecture=dict(
        type='MMSegArchitecture',
        model=student,
    ),
    distiller=dict(
        type='OSDTeacherDistiller',
        teacher=teacher,
        teacher_trainable=False,
        components=[
            dict(
                student_module='decode_head.conv_seg',
                teacher_module='decode_head.conv_seg',
                losses=[
                    dict(
                        type='KLDivergence',
                        name='loss_decode_kd_head',
                        tau=1,
                        loss_weight=1.5,
                    ),
                ]),
            dict(
                student_module='auxiliary_head.conv_seg',
                teacher_module='auxiliary_head.conv_seg',
                losses=[
                    dict(
                        type='KLDivergence',
                        name='loss_auxiliary_kd_head',
                        tau=1,
                        loss_weight=0.5,
                    ),
                ]),
        ]),

)