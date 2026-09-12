from config import CONFIG
from data.cifar import *

def test_cifar100():
    train_transform, test_transform = build_transforms(CONFIG.img_size_encoder, CONFIG.crop_pct)
    result = get_cifar100_datasets(train_transform, test_transform,
            CONFIG.path_dataset,
            num_old_classes=CONFIG.num_old_classes,
            prop_train_labels=CONFIG.prop_train_labels,
            n_views=CONFIG.n_views,
            download=CONFIG.download,
            seed=CONFIG.seed)
    
    N = 50000
    assert len(result['train_labelled']) == int(N*CONFIG.num_old_classes/100*CONFIG.prop_train_labels)
    assert len(result['train_unlabelled']) == N-int(N*CONFIG.num_old_classes/100*CONFIG.prop_train_labels)
    assert len(result['test']) == 10000

    wrapper_cifar = WrapperCIFAR(result['train_labelled'], result['train_unlabelled'])
    sampler = build_sampler(wrapper_cifar)
    
    ### A DEPLACER DANS LE TEST DU TRAIN
    train_dataloader = DataLoader(
        dataset=wrapper_cifar,
        batch_size=CONFIG.batch_size_train,
        drop_last=True,
        sampler=sampler
    )

    eval_dataloader = DataLoader(
        dataset=result['train_unlabelled_eval'],
        batch_size=CONFIG.batch_size_eval,
        drop_last=False
    )

    batch = next(iter(train_dataloader))
    views, labels, mask = batch

    assert len(views) == CONFIG.n_views
    for v in views:
        assert v.shape == (CONFIG.batch_size_train, 3, CONFIG.img_size_encoder, CONFIG.img_size_encoder)
        assert v.dtype == torch.float32

    assert labels.shape == (CONFIG.batch_size_train, )
    assert labels.dtype == torch.int64
    assert mask.shape == (CONFIG.batch_size_train, )
    assert mask.dtype == torch.bool
