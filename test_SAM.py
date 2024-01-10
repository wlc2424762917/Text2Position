import numpy as np
import cv2
from segment_anything import sam_model_registry, SamPredictor
import matplotlib.pyplot as plt

def initialize_sam_model(device, sam_model_type='vit_h', sam_checkpoint='/home/wanglichao/Text2Position/checkpoints/sam_vit_h_4b8939.pth'):
    sam = sam_model_registry[sam_model_type](checkpoint=sam_checkpoint)
    sam.to(device)
    predictor_sam = SamPredictor(sam)
    return predictor_sam

def run_sam(image_size, num_random_rounds, num_selected_points, point_coords, predictor_sam):
    best_score = 0
    print(image_size.shape)
    best_mask = np.zeros_like(image_size, dtype=bool)
    point_coords_new = np.zeros_like(point_coords)

    # (x, y) --> (y, x) ??? changed???
    point_coords_new[:, 0] = point_coords[:, 0]
    point_coords_new[:, 1] = point_coords[:, 1]

    print(point_coords_new)
    # Get only a random subsample of them for num_random_rounds times and choose the mask with highest confidence score
    for i in range(num_random_rounds):
        np.random.shuffle(point_coords_new)
        masks, scores, logits = predictor_sam.predict(
            point_coords=point_coords_new[:num_selected_points],
            point_labels=np.ones(point_coords_new[:num_selected_points].shape[0]),
            multimask_output=False,
        )
        if scores[0] > best_score:
            best_score = scores[0]
            best_mask = masks[0]

    return best_mask


# 绘制给定的点坐标
# Load the image using cv2 (Note: The path needs to be updated for your system)
image_path = r"/home/wanglichao/KITTI-360/data_2d_raw/2013_05_28_drive_0000_sync/image_00/data_rect/0000000250.png"
image = cv2.imread(image_path)
np_image2D = image[:,:,::-1]
# Display the image
plt.imshow(image)

points = np.array([
    [1385, 158], [1382, 156], [1395, 174], [1381, 157], [1386, 157], [1384, 162], [1406, 149], [1391, 152],
    [1405, 156], [1383, 158], [1389, 153], [1388, 161], [1384, 159], [1392, 167], [1405, 147], [1389, 157],
    [1390, 164], [1390, 152], [1394, 150], [1384, 161], [1392, 166], [1391, 158], [1384, 160], [1391, 160],
    [1383, 157], [1394, 151], [1385, 154], [1392, 150], [1383, 156], [1386, 155], [1389, 154], [1392, 153],
    [1394, 149], [1398, 172], [1387, 153], [1396, 152], [1392, 154], [1394, 153], [1385, 156], [1404, 147],
    [1380, 157], [1385, 157], [1404, 149], [1392, 159], [1400, 172], [1397, 172], [1392, 152], [1392, 165],
    [1406, 147], [1392, 157], [1407, 146], [1384, 156], [1392, 155], [1395, 150], [1391, 163]
])

plt.scatter(points[:, 0], points[:, 1])
plt.show()

predictor_sam = initialize_sam_model(device='cuda')
predictor_sam.set_image(np_image2D)
best_mask = run_sam(image_size=np_image2D,
                    num_random_rounds=1,
                    num_selected_points=15,
                    point_coords=points,
                    predictor_sam=predictor_sam,)

plt.figure()
plt.imshow(best_mask, alpha=0.5)
plt.show()



