# dataset settings
data = dict(
    dataset_type='NUDT',
    data_root='/root/autodl-tmp/NUDT',
    base_size=256,
    crop_size=256,
    data_aug=True,
    suffix='png',
    num_workers=8,
    train_batch=8,
    test_batch=8,
    train_dir='trainval',
    test_dir='test'
)