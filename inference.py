import cv2
import time
import torch
import argparse
import json
from PIL import Image
import torchvision.transforms as transforms
from torchvision.utils import save_image
import numpy as np

from modeling.completion_network import CompletionNetwork
from utils.poisson_blending import poisson_blend
from utils.generate_random_holes import gen_input_mask


# mouse callback function
def draw_circle(event, x, y, flags, param):
    global ix, iy, drawing, mode

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            if mode:
                cv2.rectangle(img, (ix, iy), (x, y), (0, 255, 0), -1)
                cv2.rectangle(mask, (ix, iy), (x, y), (255, 255, 255), -1)
            else:
                cv2.circle(img, (x, y), 5, (0, 0, 255), -1)
                cv2.circle(mask, (x, y), 5, (255, 255, 255), -1)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        if mode:
            cv2.rectangle(img, (ix, iy), (x, y), (0, 255, 0), -1)
            cv2.rectangle(mask, (ix, iy), (x, y), (255, 255, 255), -1)
        else:
            cv2.circle(img, (x, y), 5, (0, 0, 255), -1)
            cv2.circle(mask, (x, y), 5, (255, 255, 255), -1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_img', default='test_imgs/test_3.jpg')
    parser.add_argument('--output_img', default='test_imgs/test_3_output.jpg')
    parser.add_argument('--model', default='weights/completion/completion_weights.pth')
    parser.add_argument('--config', default='config.json')
    parser.add_argument('--mode', default='manual', choices=['manual', 'random'])
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = json.load(f)
    mpv = torch.tensor(config['mpv']).view(1, 3, 1, 1)
    model = CompletionNetwork()
    model = torch.nn.DataParallel(model)
    model.load_state_dict(torch.load(args.model, map_location='cpu'))

    img = Image.open(args.input_img)
    img = transforms.Resize(160)(img)
    img = transforms.RandomCrop((160, 160))(img)
    x = transforms.ToTensor()(img)
    x = torch.unsqueeze(x, dim=0)

    if args.mode == 'manual':
        mask = np.zeros((img.size[1], img.size[0], 3), np.uint8)
        drawing = False  # true if mouse is pressed
        mode = True  # if True, draw rectangle. Press 'm' to toggle to curve
        ix, iy = -1, -1

        img = cv2.imread(args.input_img)
        img = cv2.resize(img, (160, 160))
        cv2.namedWindow('image')
        cv2.setMouseCallback('image', draw_circle)

        while True:
            cv2.imshow('image', img)
            # cv2.imshow('mask', mask)
            k = cv2.waitKey(1) & 0xFF
            if k == ord('m'):
                mode = not mode
            elif k == 27:
                break

        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        mask = transforms.ToTensor()(mask).unsqueeze(0)

        cv2.destroyAllWindows()

    else:
        mask = gen_input_mask(shape=(1, 1, x.shape[2], x.shape[3]), hole_size=((25, 50), (25, 50),), max_holes=3)

    with torch.no_grad():
        x_mask = x - x * mask + mpv * mask
        input = torch.cat((x_mask, mask), dim=1)
        output = model(input)
        inpainted = poisson_blend(x, output, mask)
        imgs = torch.cat((x, x_mask, inpainted), dim=0)
        save_image(imgs, args.output_img, nrow=3)
    print('inference done. See result at %s.' % args.output_img)

    time.sleep(0.5)
    result_img = cv2.imread(args.output_img)
    cv2.imshow('result', result_img)
    cv2.waitKey()
