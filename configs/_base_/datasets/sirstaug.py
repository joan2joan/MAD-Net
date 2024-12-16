# dataset settings
data = dict(
    dataset_type='SIRSTAUG',
    data_root='/root/autodl-tmp/SIRST_AUG',
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
