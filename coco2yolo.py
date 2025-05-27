from ultralytics.data.converter import convert_coco

convert_coco(labels_dir="_data/annotations/", use_segments=True)