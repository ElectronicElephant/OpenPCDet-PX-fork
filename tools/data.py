from pcdet.datasets.sensetime.sensetime_dataset import create_sensetime_infos



if __name__ == '__main__':
    import sys
    if sys.argv.__len__() > 1 and sys.argv[1] == 'create_sensetime_infos':
        import yaml
        from pathlib import Path
        from easydict import EasyDict
        dataset_cfg = EasyDict(yaml.load(open(sys.argv[2])))
        ROOT_DIR = Path(dataset_cfg.DATA_PATH).resolve()
        create_sensetime_infos(
            dataset_cfg=dataset_cfg,
            class_names=['Vehicle', 'Pedestrian', 'Cyclist', 'Bus'],
            data_path=ROOT_DIR,
            save_path=ROOT_DIR / 'sensetime'
        )