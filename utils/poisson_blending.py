import torch
import torchvision.transforms as transforms
import numpy as np
import cv2


def poisson_blend(x, output, mask):
    x = x.clone().cpu()
    output = output.clone().cpu()
    mask = mask.clone().cpu()
    mask = torch.cat((mask, mask, mask), dim=1)
    num_samples = x.shape[0]
    ret = []

    for i in range(num_samples):
        transted_img = transforms.functional.to_pil_image(x[i])
        transted_img = np.array(transted_img)[:, :, [2, 1, 0]]
        src_img = transforms.functional.to_pil_image(output[i])
        src_img = np.array(src_img)[:, :, [2, 1, 0]]
        msk = transforms.functional.to_pil_image(mask[i])
        msk = np.array(msk)[:, :, [2, 1, 0]]

        xs, ys = [], []
        for i in range(msk.shape[0]):
            for j in range(msk.shape[1]):
                if msk[i, j, 0] == 255:
                    ys.append(i)
                    xs.append(j)

        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        
        center = ((x_max + x_min) // 2, (y_max + y_min) // 2)
        out = cv2.seamlessClone(src_img, transted_img, msk, center, cv2.NORMAL_CLONE)
        out = out[:, :, [2, 1, 0]]
        out = transforms.functional.to_tensor(out)
        out = torch.unsqueeze(out, dim=0)
        ret.append(out)

    ret = torch.cat(ret, dim=0)
    return ret
