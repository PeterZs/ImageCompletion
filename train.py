import logging
from tqdm import tqdm
import torch.optim
from torch.nn import DataParallel, BCELoss

from modeling.completion_network import CompletionNetwork
from modeling.context_discriminator import ContextDiscriminator
from data_loader.data_loader import train_data_loader, test_data_loader, sample_random_batch
from utils.add_mask import add_mask, get_train_mean
from utils.poisson_blending import poisson_blend
from losses.losses import completion_network_loss
from utils.generate_random_holes import *

from opt import opt

from torchvision.utils import save_image
from PIL import Image
import torchvision.transforms as transforms
import random
import os
import argparse
import numpy as np
import json
import time


def train_phase1(opt, device):
    completion_net = CompletionNetwork()
    if torch.cuda.device_count() > 1:
        logging.info("It will use", torch.cuda.device_count(), "GPUs!")
        completion_net = DataParallel(completion_net)
    if opt.weight_completion is not None:
        completion_net.load_state_dict(torch.load(opt.weight_completion, map_location='cpu'))
    completion_net = completion_net.to(device)
    # get optimizer
    optimizer = torch.optim.Adadelta(completion_net.parameters())

    # begin to train
    process_bar = tqdm(total=opt.iteration1)
    while process_bar.n < opt.iteration1:
        for batch in train_loader:
            # forward
            batch = batch.to(device)

            hole_area = gen_hole_area((opt.ld_input_size, opt.ld_input_size), (batch.shape[3], batch.shape[2]))
            mask, input_batch, _ = add_mask(opt, device, batch, hole_area, mpv)
            output_batch = completion_net(input_batch)
            loss = completion_network_loss(batch, output_batch, mask)

            # backward
            loss.backward()

            # optimize
            optimizer.step()
            optimizer.zero_grad()
            process_bar.set_description('phase 1 | train loss: %.5f' % loss.cpu())
            process_bar.update()

            # # test
            #
            if process_bar.n % opt.test_period1 == 0:
                with torch.no_grad():
                    test_batch = sample_random_batch(test_data_set, batch_size=opt.test_batch_size).to(device)
                    hole_area = gen_hole_area((opt.ld_input_size, opt.ld_input_size), (batch.shape[3], batch.shape[2]))
                    mask, input_batch, batch_data_with_mask = add_mask(opt, device, test_batch, hole_area, mpv)
                    output_batch = completion_net(input_batch)
                    completed = poisson_blend(test_batch, output_batch, mask)

                    # return to cpu and save test result
                    imgs = torch.cat((test_batch.cpu(), batch_data_with_mask.cpu(), completed.cpu()), dim=0)
                    imgpath = os.path.join(opt.result_path, 'phase_1', 'step%d.png' % process_bar.n)
                    save_image(imgs, imgpath, nrow=len(test_batch))

                    # save model
                    model_cn_path = os.path.join(opt.result_path, 'phase_1', 'completion_step%d.pth' % process_bar.n)
                    torch.save(completion_net.state_dict(), model_cn_path)

            if process_bar.n >= opt.iteration1:
                break
    process_bar.close()
    return completion_net


def train_phase2(opt, device, completion_net):
    discriminator_net = ContextDiscriminator(
        local_input_shape=(3, opt.ld_input_size, opt.ld_input_size),
        global_input_shape=(3, opt.cn_input_size, opt.cn_input_size),
        arc=opt.dataset
    )

    if torch.cuda.device_count() > 1:
        discriminator_net = DataParallel(discriminator_net)
    if opt.weight_discriminator is not None:
        discriminator_net.load_state_dict(torch.load(opt.weight_discriminator, map_location='cpu'))
    discriminator_net = discriminator_net.to(device)
    # get optimizer
    optimizer = torch.optim.Adadelta(discriminator_net.parameters())

    # begin to train
    process_bar = tqdm(total=opt.iteration2)
    while process_bar.n < opt.iteration2:
        for batch in train_loader:
            # forward of fake
            batch = batch.to(device)

            hole_area_fake = gen_hole_area((opt.ld_input_size, opt.ld_input_size), (batch.shape[3], batch.shape[2]))
            mask = gen_input_mask(
                shape=(batch.shape[0], 1, batch.shape[2], batch.shape[3]),
                hole_size=((opt.hole_min_w, opt.hole_max_w), (opt.hole_min_h, opt.hole_max_h)),
                hole_area=hole_area_fake,
                max_holes=opt.max_holes,
            ).to(device)
            fake = torch.zeros((len(batch), 1)).to(device)
            x_mask = batch - batch * mask + mpv * mask
            input_cn = torch.cat((x_mask, mask), dim=1)
            output_cn = completion_net(input_cn)
            input_gd_fake = output_cn.detach()
            input_ld_fake = crop(input_gd_fake, hole_area_fake)
            output_fake = discriminator_net((input_ld_fake.to(device), input_gd_fake.to(device)))
            bce_loss = BCELoss()
            loss_fake = bce_loss(output_fake, fake)

            # real forward
            hole_area_real = gen_hole_area(size=(opt.ld_input_size, opt.ld_input_size),
                                           mask_size=(batch.shape[3], batch.shape[2]))
            real = torch.ones((len(batch), 1)).to(device)
            input_gd_real = batch
            input_ld_real = crop(input_gd_real, hole_area_real)
            output_real = discriminator_net((input_ld_real, input_gd_real))
            loss_real = bce_loss(output_real, real)

            # reduce
            loss = (loss_fake + loss_real) / 2.

            # backward
            loss.backward()

            # optimize
            optimizer.step()
            optimizer.zero_grad()
            process_bar.set_description('phase 2 | train loss: %.5f' % loss.cpu())
            process_bar.update()

            # # test
            if process_bar.n % opt.test_period2 == 0:
                with torch.no_grad():
                    model_cn_path = os.path.join(opt.result_path, 'phase_2', 'discriminator_step%d.pth' % process_bar.n)
                    torch.save(discriminator_net.state_dict(), model_cn_path)

            if process_bar.n >= opt.iteration2:
                break
    process_bar.close()
    return discriminator_net


def train_phase3(opt, device, completion_net, discriminator_net):
    # set optimizer
    completion_optimizer = torch.optim.Adadelta(completion_net.parameters())
    discriminator_optimizer = torch.optim.Adadelta(discriminator_net.parameters())

    process_bar = tqdm(total=opt.iteration3)
    while process_bar.n < opt.iteration3:
        for batch in train_loader:
            # forward of fake
            batch = batch.to(device)

            hole_area_fake = gen_hole_area((opt.ld_input_size, opt.ld_input_size), (batch.shape[3], batch.shape[2]))
            mask, completion_input, _ = add_mask(opt, device, batch, hole_area_fake, mpv)
            completion_output = completion_net(completion_input)

            # input for fake
            input_gd_fake = completion_output.detach()
            input_ld_fake = crop(input_gd_fake, hole_area_fake)
            output_fake = discriminator_net((input_ld_fake.to(device), input_gd_fake.to(device)))
            fake = torch.zeros((len(batch), 1)).to(device)
            bce_loss = BCELoss()
            loss_fake = bce_loss(output_fake, fake)

            # real forward
            hole_area_real = gen_hole_area(size=(opt.ld_input_size, opt.ld_input_size),
                                           mask_size=(batch.shape[3], batch.shape[2]))
            real = torch.ones((len(batch), 1)).to(device)
            input_gd_real = batch
            input_ld_real = crop(input_gd_real, hole_area_real)
            output_real = discriminator_net((input_ld_real, input_gd_real))
            loss_real = bce_loss(output_real, real)

            discriminator_loss = (loss_fake + loss_real) * alpha / 2.
            discriminator_loss.backward()
            discriminator_optimizer.step()
            discriminator_optimizer.zero_grad()

            # backward completion network
            completion_loss1 = completion_network_loss(batch, completion_output, mask)
            input_gd_fake = completion_output
            input_ld_fake = crop(input_gd_fake, hole_area_fake)
            output_fake = discriminator_net((input_ld_fake, input_gd_fake))
            completion_loss2 = bce_loss(output_fake, real)
            completion_loss = (completion_loss1 + alpha * completion_loss2) / 2.
            completion_loss.backward()
            completion_optimizer.step()
            completion_optimizer.zero_grad()

            process_bar.set_description('phase 3 | train loss (cd): %.5f (cn): %.5f' % (discriminator_loss.cpu(), completion_loss.cpu()))
            process_bar.update()

            # # test
            if process_bar.n % opt.test_period3 == 0:
                with torch.no_grad():
                    test_batch = sample_random_batch(test_data_set, batch_size=opt.test_batch_size).to(device)
                    hole_area = gen_hole_area((opt.ld_input_size, opt.ld_input_size), (batch.shape[3], batch.shape[2]))
                    mask, input_batch, batch_data_with_mask = add_mask(opt, device, test_batch, hole_area, mpv)
                    output_batch = completion_net(input_batch)
                    completed = poisson_blend(test_batch, output_batch, mask)

                    # return to cpu and save test result
                    imgs = torch.cat((test_batch.cpu(), batch_data_with_mask.cpu(), completed.cpu()), dim=0)
                    imgpath = os.path.join(opt.result_path, 'phase_3', 'step%d.png' % process_bar.n)
                    save_image(imgs, imgpath, nrow=len(test_batch))

                    # save model
                    model_cn_path = os.path.join(opt.result_path, 'phase_3', 'completion_step%d.pth' % process_bar.n)
                    torch.save(completion_net.state_dict(), model_cn_path)
                    model_cd_path = os.path.join(opt.result_path, 'phase_3', 'discriminator_step%d.pth' % process_bar.n)
                    torch.save(discriminator_net.state_dict(), model_cd_path)

            if process_bar.n >= opt.iteration3:
                break
    process_bar.close()


if __name__ == '__main__':
    # choose device
    device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")

    # create result directory (if necessary)
    if not os.path.exists(opt.result_path):
        os.makedirs(opt.result_path)
    for s in ['phase_1', 'phase_2', 'phase_3']:
        if not os.path.exists(os.path.join(opt.result_path, s)):
            os.makedirs(os.path.join(opt.result_path, s))

    # get train and test data
    train_data_set, train_loader = train_data_loader(opt)
    test_data_set, test_loader = test_data_loader(opt)
    # calculate mvp
    alpha = torch.tensor(opt.alpha).to(device)
    mpv = get_train_mean(opt, train_data_set, device)

    # begin to train
    completion_net = train_phase1(opt, device)
    discriminator_net = train_phase2(opt, device, completion_net)
    train_phase3(opt, device, completion_net, discriminator_net)
