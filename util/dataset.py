from .collective import *
# from .volleyball import VolleyballDataset, volley_read_dataset, volley_all_frames
from .volleybll import volleyball_read_labelannotations, volleyball_all_frames, VolleyballDataset


def returnDataset(cfg):
    if cfg.dataset.data_name == 'collective':
        train_anns = collective_read_dataset(cfg.dataset.data_path, cfg.dataset.train.seqs)
        train_frames = collective_all_frames(train_anns)

        train_frames = 6*train_frames
        test_anns = collective_read_dataset(cfg.dataset.data_path, cfg.dataset.val.seqs)
        test_frames = collective_all_frames(test_anns)
        training_set = CollectiveDataset(train_anns, train_frames,
                                         cfg.dataset.data_path, cfg.dataset.image_size, cfg.dataset.out_size,
                                         num_frames=cfg.dataset.num_frames, is_training=True,
                                         is_finetune=False)

        validation_set = CollectiveDataset(test_anns, test_frames,
                                           cfg.dataset.data_path, cfg.dataset.image_size, cfg.dataset.out_size,
                                           num_frames=cfg.dataset.num_frames, is_training=False,
                                           is_finetune=False)

    if cfg.dataset.data_name == 'volleyball':
        train_anns = volleyball_read_labelannotations(cfg.dataset.train.data_path, cfg.dataset.train.seqs)
        train_frames = volleyball_all_frames(train_anns)

        test_anns = volleyball_read_labelannotations(cfg.dataset.val.data_path, cfg.dataset.val.seqs)
        test_frames = volleyball_all_frames(test_anns)

        all_anns = {**train_anns, **test_anns}

        training_set = VolleyballDataset(all_anns, train_frames,
                                         cfg.dataset.train.data_path,cfg.dataset.image_size,cfg.dataset.out_size, num_before=cfg.dataset.num_before,
                                         num_after=cfg.dataset.num_after, is_training=True,
                                         is_finetune=True)

        validation_set = VolleyballDataset(all_anns, test_frames,
                                           cfg.dataset.val.data_path, cfg.dataset.image_size, cfg.dataset.out_size, num_before=cfg.dataset.num_before,
                                           num_after=cfg.dataset.num_after, is_training=False,
                                           is_finetune=True)

    print('Reading dataset finished...')
    print('%d train samples' % len(train_frames))
    print('%d test samples' % len(test_frames))
    return training_set, validation_set
