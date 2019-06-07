from torch.nn.functional import mse_loss


def completion_network_loss(input_img, output_img, mask):
    return mse_loss(output_img * mask, input_img * mask)
