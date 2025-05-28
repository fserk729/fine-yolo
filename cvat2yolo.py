import os
import sys
import cv2
import yaml
import numpy as np
import xml.etree.ElementTree as ET
from scipy.spatial import distance
from tqdm import tqdm
from pathlib import Path
 
 
def mask_to_polygon(mask, min_area=20):
    polygons = []
    cntrs, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    for cnt in cntrs:
        cnt = cnt[:, 0, :]
        if cv2.contourArea(cnt) >= min_area:
            polygons.append(cnt)
    while len(polygons) > 1:
        poly1 = polygons.pop()
        closest_poly_dist = np.inf
        closest_poly_idx = None
        closest_points_idx = None
        for i, p in enumerate(polygons):
            dists = distance.cdist(poly1, p)
            min_dist_idx = np.unravel_index(dists.argmin(), dists.shape)
            if dists[min_dist_idx] < closest_poly_dist:
                closest_poly_dist = dists[min_dist_idx]
                closest_poly_idx = i
                closest_points_idx = min_dist_idx
        poly2 = polygons.pop(closest_poly_idx)
        new_poly = np.concatenate((np.roll(poly1, -closest_points_idx[0] - 1, axis=0),
                                   np.roll(poly2, -closest_points_idx[1], axis=0)))
        polygons.append(new_poly)
    return np.array(polygons.pop(), dtype=np.int32)
 
 
def long_path(path):
    abspath = os.fspath(path.resolve())
    if os.name == 'nt' and len(abspath) > 256:
        return Path('\\\\?\\' + abspath)
    return path
 
 
def cvat_to_yolo(xml_path):
    try:
        tree = ET.parse(xml_path)
    except ET.ParseError:
        print(f"❌ Fout bij het parsen van XML-bestand: {xml_path}")
        return 1
    except FileNotFoundError:
        print(f"❌ Bestand niet gevonden: {xml_path}")
        return 1
 
    root = tree.getroot()
    meta = root.find('meta')
    if meta is None:
        print("⚠️  'meta' element niet gevonden in XML. Controleer exportformaat.")
        return 1
 
    # Fallback voor project- of task-gebaseerde export
    labels = None
    if meta.find('project') is not None:
        labels = meta.find('project').find('labels')
    elif meta.find('task') is not None:
        labels = meta.find('task').find('labels')
 
    if labels is None:
        print("⚠️  'labels' element niet gevonden in 'meta'.")
        return 1
 
    # Maak label mapping
    label_dict = {}
    for i, label in enumerate(labels.findall('label')):
        label_name = label.find('name').text
        label_dict[label_name] = i
 
    # Verwerk maskers per afbeelding
    for img in tqdm(root.findall('image')):
        img_name = img.get('name')
        img_h = int(img.get('height'))
        img_w = int(img.get('width'))
        lines = []
 
        for mask in img.findall('mask'):
            mask_label = mask.get('label')
            mask_rle = [int(val.strip()) for val in mask.get('rle').split(',')]
            mask_top = int(mask.get('top'))
            mask_left = int(mask.get('left'))
            mask_h = int(mask.get('height'))
            mask_w = int(mask.get('width'))
 
            decoded = [0] * (mask_h * mask_w)
            idx = 0
            value = 0
            for xy in mask_rle:
                decoded[idx:idx + xy] = [value] * xy
                idx += xy
                value = abs(value - 1)
 
            decoded = np.asarray(decoded, dtype=np.uint8).reshape((mask_h, mask_w))
            full_mask = np.zeros((img_h, img_w), dtype=np.uint8)
            full_mask[mask_top:mask_top + mask_h, mask_left:mask_left + mask_w] = decoded
 
            poly = mask_to_polygon(full_mask)
            line = [str(label_dict[mask_label])]
            for point in poly:
                line.append(str(round(point[0] / img_w, 5)))
                line.append(str(round(point[1] / img_h, 5)))
            lines.append(' '.join(line))
 
        if lines:
            img_path = Path(img_name)
            label_path = Path("labels") / img_path.parent / (img_path.stem + '.txt')
            label_path = long_path(label_path)
            label_path.parent.mkdir(parents=True, exist_ok=True)
            label_path.write_text('\n'.join(lines))
 
    # YAML maken
    with open('dataset.yaml', 'w') as yaml_file:
        yaml.dump({'names': label_dict}, yaml_file, sort_keys=False)
 
    print("✅ Conversie voltooid. YOLO-labels opgeslagen in map: labels/")
    return 0
 
 
if __name__ == '__main__':
    try:
        annotations = sys.argv[1]
    except IndexError:
        print('Gebruik: python cvat_to_yolo.py <pad/naar/annotations.xml>')
        sys.exit(1)
 
    sys.exit(cvat_to_yolo(annotations))