import os
from multiprocessing.pool import Pool

import cv2
import numpy
import warnings
from tqdm.auto import tqdm

try:
    from detectron2.structures import BoxMode
    bbox_mode = BoxMode.XYXY_ABS
except ImportError:
    warnings.warn(
        'detectron2 not installed, assuming value 0 for '
        'detectron2.structures.BoxMode.XYXY_ABS, '
        'this will unlikely cause issues, but installing '
        'detectron2 and running again would be safer.', RuntimeWarning)
    bbox_mode = 0
from syntheticyeastcells import create_samples

def get_annotation(label):
    contours, _ = cv2.findContours(label.astype(numpy.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return {"bbox": [0, 0, 0, 0], "bbox_mode": bbox_mode, "segmentation": [], "category_id": 0, "iscrowd": 0}
    contour = max(contours, key=len)
    contour = [p_ for p in contour[:, 0, :] + 0.5 for p_ in p]
    contour += contour[:2]
    px, py = contour[::2], contour[1::2]

    return {
        "bbox": [numpy.min(px), numpy.min(py), numpy.max(px), numpy.max(py)],
        "bbox_mode": bbox_mode,
        "segmentation": [contour],
        "category_id": 0,
        "iscrowd": 0
    }

def get_annotations(label):
    return {
        'height': label.shape[0],
        'width': label.shape[1],
        'annotations': [
            get_annotation(label == i)
            for i in range(1, label.max() + 1)]}

def process_batch(destination, set_name, start, end,
                  n_cells_per_image=100,
                  size=(512, 512),
                  min_distance_boundary=50,    # minimum distance of center from image boundary
                  r0_range=(2, 14),            # range of the first radius
                  r1_factor_range=(0.7, 1.3),  # range of the second radius as a factor of the first.
                  spatial_blur_std=1.5,
                  background_intensity=0.4,
                  background_contrast=0.00188,
                  core_contrast=0.0752,
                  vac_contrast=1,
                  p_white_outside=0.5,
                  k=1,
                  x0=0,
                  strictness='normal',
                  bud_cells=0,
                  cell_bud_ratio=4,
                  include_vacuoles=True
                 ):
    os.makedirs(f'{destination}/{set_name}/', exist_ok=True)
    left = [
        (i, fn) for i, fn in {
#             (i, f'{destination}/{set_name}/image-{i}.jpg')
            (i, f'{set_name}/image-{i}.jpg')
            for i in range(start, end)
        }
        # if not os.path.exists(fn)
    ]
    images, labels = create_samples(
       len(left),
       n_cells_per_image=n_cells_per_image, size=size,
       min_distance_boundary=min_distance_boundary,
       r0_range=r0_range,
       r1_factor_range=r1_factor_range,
       spatial_blur_std=spatial_blur_std,
       background_intensity=background_intensity,
       background_contrast=background_contrast,
       core_contrast=core_contrast,
       vac_contrast=vac_contrast,
       p_white_outside=p_white_outside,
       k=k,
       x0=x0,
       strictness=strictness,
       bud_cells=bud_cells,
       cell_bud_ratio=cell_bud_ratio,
       include_vacuoles=include_vacuoles)

    data = []
    for (i, filename), label, image in zip(left, labels, images):
#         cv2.imwrite(filename, image)
        cv2.imwrite(f'{destination}{filename}', image)
        data.append(get_annotations(label))
        data[-1]['file_name'] = filename
        data[-1]['image_id'] = f'{set}-{i:05d}'
    return data

def create_dataset(destination,
                   sets={'test': 1000, 'train': 20000, 'val': 1000},
                   n_cells_per_image=100,
                   size=(512, 512),
                   min_distance_boundary=50,    # minimum distance of center from image boundary
                   r0_range=(2, 14),            # range of the first radius
                   r1_factor_range=(0.7, 1.3),  # range of the second radius as a factor of the first.
                   spatial_blur_std=1.5,
                   background_intensity=0.4,
                   background_contrast=0.00188,
                   core_contrast=0.0752,
                   vac_contrast=1,
                   p_white_outside=0.5,
                   k=1,
                   x0=0,
                   strictness='normal',
                   bud_cells = 0,
                   cell_bud_ratio = 4,
                   include_vacuoles = True,
                   njobs=40, batch_size=10,
                   progressbar=True):
    kwargs = {
        'n_cells_per_image': n_cells_per_image, 'size': size,
        'min_distance_boundary': min_distance_boundary,
        'r0_range': r0_range,
        'r1_factor_range': r1_factor_range,
        'spatial_blur_std': spatial_blur_std,
        'background_intensity': background_intensity,
        'background_contrast': background_contrast,
        'core_contrast': core_contrast,
        'vac_contrast': vac_contrast,
        'p_white_outside': p_white_outside,
        'k': k,
        'x0': x0,
        'strictness': strictness,
        'bud_cells': bud_cells,
        'cell_bud_ratio': cell_bud_ratio,
        'include_vacuoles': include_vacuoles}
    progressbar = tqdm if progressbar else (lambda x: x)

    results = dict()
    with Pool(njobs) as pool:
        for set_, n in sets.items():
            results[set_] = [
                pool.apply_async(process_batch,
                                 [destination, set_, start, end],
                                 kwargs)
                for start in range(0, n, batch_size)
                for end in [min(n, start + batch_size)]  # alias
            ]

        result = {}
        for k, r in progressbar([
            (k, r) for k, v in results.items() for r in v]):
            result[k] = result.get(k, [])
            result[k].extend(r.get())

    return result
