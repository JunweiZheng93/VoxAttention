import matplotlib.pyplot as plt
import os


def stack_swapped_plot(input_img_dir,
                       H_crop_factor=0.2,
                       W_crop_factor=0.55,
                       H_shift=15,
                       W_shift=40):
    images = sorted(os.listdir(input_img_dir))
    gt_list = list()
    ori_recon_list = list()
    ori_swapped_list = list()
    attention_recon_list = list()
    attention_swap_list = list()
    for img in images:
        if 'gt' in img:
            gt_list.append(os.path.join(input_img_dir, img))
        elif 'ori' in img and 'recon' in img:
            ori_recon_list.append(os.path.join(input_img_dir, img))
        elif 'ori' in img and 'swap' in img:
            ori_swapped_list.append(os.path.join(input_img_dir, img))
        elif 'attention' in img and 'recon' in img:
            attention_recon_list.append(os.path.join(input_img_dir, img))
        else:
            attention_swap_list.append(os.path.join(input_img_dir, img))

    # every image has H=240, W=320 pixels
    # crop every single images to get a bette view
    # because it may have too much empty place around the full shape
    H_img_start = int(240*H_crop_factor/4) + H_shift
    W_img_start = int(320*W_crop_factor/4) + W_shift
    H_img_end = int(H_img_start+240*(1-H_crop_factor))
    W_img_end = int(W_img_start+320*(1-W_crop_factor))
    H_fig = 2.4*(1-H_crop_factor)*len(gt_list)
    W_fig = 3.2*(1-W_crop_factor)*5  # 5 images per row

    fig = plt.figure(figsize=(W_fig, H_fig))
    for i in range(len(gt_list)):
        for j in range(5):
            ax = fig.add_axes([j/5, 1-((1+i)/len(gt_list)), 1/5, 1/len(gt_list)])
            ax.axis('off')
            if j == 0:
                ax.imshow(plt.imread(gt_list[i])[H_img_start:H_img_end, W_img_start:W_img_end])
            elif j == 1:
                ax.imshow(plt.imread(attention_recon_list[i])[H_img_start:H_img_end, W_img_start:W_img_end])
            elif j == 2:
                ax.imshow(plt.imread(attention_swap_list[i])[H_img_start:H_img_end, W_img_start:W_img_end])
            elif j == 3:
                ax.imshow(plt.imread(ori_recon_list[i])[H_img_start:H_img_end, W_img_start:W_img_end])
            else:
                ax.imshow(plt.imread(ori_swapped_list[i])[H_img_start:H_img_end, W_img_start:W_img_end])

    plt.savefig(os.path.join(input_img_dir, 'stacked_image.png'))


def stack_interpolation_plot(input_img_dir,
                             H_crop_factor=0.2,
                             W_crop_factor=0.55,
                             H_shift=15,
                             W_shift=40):
    images = sorted(os.listdir(input_img_dir))
    gt_list = list()
    interpolation_list = list()
    for img in images:
        if 'gt' in img:
            gt_list.append(os.path.join(input_img_dir, img))
        else:
            interpolation_list.append(os.path.join(input_img_dir, img))

    # every image has H=240, W=320 pixels
    # crop every single images to get a bette view
    # because it may have too much empty place around the full shape
    H_img_start = int(240*H_crop_factor/4) + H_shift
    W_img_start = int(320*W_crop_factor/4) + W_shift
    H_img_end = int(H_img_start+240*(1-H_crop_factor))
    W_img_end = int(W_img_start+320*(1-W_crop_factor))
    H_fig = 2.4*(1-H_crop_factor)*len(gt_list)/2
    W_fig = 3.2*(1-W_crop_factor)*10  # 10 images per row

    fig = plt.figure(figsize=(W_fig, H_fig))
    for i in range(int(len(gt_list)/2)):
        for j in range(10):
            ax = fig.add_axes([j/10, 1-((1+i)/len(gt_list)*2), 1/10, 1/len(gt_list)*2])
            ax.axis('off')
            if j == 0:
                ax.imshow(plt.imread(gt_list[0+2*i])[H_img_start:H_img_end, W_img_start:W_img_end])
            elif j == 9:
                ax.imshow(plt.imread(gt_list[1+2*i])[H_img_start:H_img_end, W_img_start:W_img_end])
            else:
                ax.imshow(plt.imread(interpolation_list[j-1+8*i])[H_img_start:H_img_end, W_img_start:W_img_end])

    plt.savefig(os.path.join(input_img_dir, 'stacked_image.png'))


def stack_mix_plot(input_img_dir,
                   H_crop_factor=0.2,
                   W_crop_factor=0.55,
                   H_shift=15,
                   W_shift=40):
    images = sorted(os.listdir(input_img_dir))
    gt_list = list()
    mixed_list = list()
    for img in images:
        if 'gt' in img:
            gt_list.append(os.path.join(input_img_dir, img))
        else:
            mixed_list.append(os.path.join(input_img_dir, img))

    # every image has H=240, W=320 pixels
    # crop every single images to get a bette view
    # because it may have too much empty place around the full shape
    H_img_start = int(240 * H_crop_factor / 4) + H_shift
    W_img_start = int(320 * W_crop_factor / 4) + W_shift
    H_img_end = int(H_img_start + 240 * (1 - H_crop_factor))
    W_img_end = int(W_img_start + 320 * (1 - W_crop_factor))
    H_fig = 2.4 * (1 - H_crop_factor) * 2  # 2 images per column
    W_fig = 3.2 * (1 - W_crop_factor) * len(gt_list)

    fig = plt.figure(figsize=(W_fig, H_fig))
    for i in range(2):
        for j in range(len(gt_list)):
            ax = fig.add_axes([j / len(gt_list), (1 - i) / 2, 1 / len(gt_list), 1 / 2])
            ax.axis('off')
            if i == 0:
                ax.imshow(plt.imread(gt_list[j])[H_img_start:H_img_end, W_img_start:W_img_end])
            else:
                ax.imshow(plt.imread(mixed_list[j])[H_img_start:H_img_end, W_img_start:W_img_end])

    plt.savefig(os.path.join(input_img_dir, 'stacked_image.png'))


def stack_reconstruction_plot(input_img_dir,
                              H_crop_factor=0.2,
                              W_crop_factor=0.55,
                              H_shift=15,
                              W_shift=40):
    images = sorted(os.listdir(input_img_dir))
    gt_list = list()
    ori_p2_list = list()
    ori_p3_list = list()
    notkeepC_p2_list = list()
    notkeepC_p3_list = list()
    keepC_p2_list = list()
    keepC_p3_list = list()
    for img in images:
        if 'gt' in img:
            gt_list.append(os.path.join(input_img_dir, img))
        elif 'ori' in img and 'p2' in img:
            ori_p2_list.append(os.path.join(input_img_dir, img))
        elif 'ori' in img and 'p3' in img:
            ori_p3_list.append(os.path.join(input_img_dir, img))
        elif 'notkeepC' in img and 'p2' in img:
            notkeepC_p2_list.append(os.path.join(input_img_dir, img))
        elif 'notkeepC' in img and 'p3' in img:
            notkeepC_p3_list.append(os.path.join(input_img_dir, img))
        elif 'keepC' in img and 'p2' in img:
            keepC_p2_list.append(os.path.join(input_img_dir, img))
        elif 'keepC' in img and 'p3' in img:
            keepC_p3_list.append(os.path.join(input_img_dir, img))

    # every image has H=240, W=320 pixels
    # crop every single images to get a bette view
    # because it may have too much empty place around the full shape
    H_img_start = int(240 * H_crop_factor / 4) + H_shift
    W_img_start = int(320 * W_crop_factor / 4) + W_shift
    H_img_end = int(H_img_start + 240 * (1 - H_crop_factor))
    W_img_end = int(W_img_start + 320 * (1 - W_crop_factor))
    H_fig = 2.4 * (1 - H_crop_factor) * len(gt_list)
    W_fig = 3.2 * (1 - W_crop_factor) * 7

    fig = plt.figure(figsize=(W_fig, H_fig))
    for i in range(len(gt_list)):
        for j in range(7):
            ax = fig.add_axes([j/7, 1-((1+i)/len(gt_list)), 1/7, 1/len(gt_list)])
            ax.axis('off')
            if j == 0:
                ax.imshow(plt.imread(gt_list[i])[H_img_start:H_img_end, W_img_start:W_img_end])
            elif j == 1:
                ax.imshow(plt.imread(ori_p2_list[i])[H_img_start:H_img_end, W_img_start:W_img_end])
            elif j == 2:
                ax.imshow(plt.imread(ori_p3_list[i])[H_img_start:H_img_end, W_img_start:W_img_end])
            elif j == 3:
                ax.imshow(plt.imread(notkeepC_p2_list[i])[H_img_start:H_img_end, W_img_start:W_img_end])
            elif j == 4:
                ax.imshow(plt.imread(notkeepC_p3_list[i])[H_img_start:H_img_end, W_img_start:W_img_end])
            elif j == 5:
                ax.imshow(plt.imread(keepC_p2_list[i])[H_img_start:H_img_end, W_img_start:W_img_end])
            else:
                ax.imshow(plt.imread(keepC_p3_list[i])[H_img_start:H_img_end, W_img_start:W_img_end])

    plt.savefig(os.path.join(input_img_dir, 'stacked_image.png'))
