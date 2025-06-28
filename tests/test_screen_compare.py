import importlib
from PIL import Image


def test_compare_images_detects_box():
    mod = importlib.import_module('see.screen_compare')
    img1 = Image.new('RGB', (100, 100), 'white')
    img2 = Image.new('RGB', (100, 100), 'white')
    for x in range(20, 40):
        for y in range(20, 40):
            img2.putpixel((x, y), (0, 0, 0))
    boxes = mod.compare_images(img1, img2, block_size=10, ssim_threshold=0.95)
    assert boxes
    box = boxes[0]['box_2d']
    assert box[0] <= 20 <= box[2]
