from torch.nn.functional import mse_loss
from torch.nn import BCELoss


def gan_loss(input1, input2):
    bce_loss = BCELoss()
    return bce_loss(input1, input2)


def completion_network_loss(input_img, output_img, mask):
    return mse_loss(output_img * mask, input_img * mask)
