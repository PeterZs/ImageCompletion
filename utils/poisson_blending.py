import torch
import random
import torchvision.transforms as transforms
import numpy as np
import cv2


def poisson_blend(x, output, mask):
    """
    * inputs:
        - x (torch.Tensor, required)
                Input image tensor of shape (N, 3, H, W).
        - output (torch.Tensor, required)
                Output tensor from Completion Network of shape (N, 3, H, W).
        - mask (torch.Tensor, required)
                Input mask tensor of shape (N, 1, H, W).
    * returns:
                An image tensor of shape (N, 3, H, W) inpainted
                using poisson image editing method.
    """
    x = x.clone().cpu()
    output = output.clone().cpu()
    mask = mask.clone().cpu()
    mask = torch.cat((mask,mask,mask), dim=1) # convert to 3-channel format
    num_samples = x.shape[0]
    ret = []
    for i in range(num_samples):
        dstimg = transforms.functional.to_pil_image(x[i])
        dstimg = np.array(dstimg)[:, :, [2, 1, 0]]
        srcimg = transforms.functional.to_pil_image(output[i])
        srcimg = np.array(srcimg)[:, :, [2, 1, 0]]
        msk = transforms.functional.to_pil_image(mask[i])
        msk = np.array(msk)[:, :, [2, 1, 0]]
        # compute mask's center
        xs, ys = [], []
        for i in range(msk.shape[0]):
            for j in range(msk.shape[1]):
                if msk[i,j,0] == 255:
                    ys.append(i)
                    xs.append(j)
        xmin, xmax = min(xs), max(xs)
        ymin, ymax = min(ys), max(ys)
        center = ((xmax + xmin) // 2, (ymax + ymin) // 2)
        out = cv2.seamlessClone(srcimg, dstimg, msk, center, cv2.NORMAL_CLONE)
        out = out[:, :, [2, 1, 0]]
        out = transforms.functional.to_tensor(out)
        out = torch.unsqueeze(out, dim=0)
        ret.append(out)
    ret = torch.cat(ret, dim=0)
    return ret