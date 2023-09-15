def build_train(cfg, mode='train'):
    if cfg.type == 'face_video':
        from .face_video import Dataset
    elif cfg.type == 'body_video':
        from .body_video import Dataset
    elif cfg.type == 'hair_video':
        from .hair_video import Dataset
    elif cfg.type == 'rp_video':
        from .rp_video import Dataset
    return Dataset.from_config(cfg, mode=mode)

