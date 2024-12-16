# dataset settings
data = dict(
    dataset_type='NUAA',
    data_root='/root/autodl-tmp/NUAA-SIRST',
    base_size=512,
    crop_size=512,
    data_aug=True,
    suffix='png',
    num_workers=8,
    train_batch=8,
    test_batch=8,
    train_dir='trainval',
    test_dir='test'
)
