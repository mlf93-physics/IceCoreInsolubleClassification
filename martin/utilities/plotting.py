import matplotlib.pyplot as plt
import seaborn_image as sb_img

def plot_images(images=None, labels=None):
    if len(images) > 20:
        raise ValueError('ERROR: Too many images to plot. Max = 20')
    images = [image.permute(1, 2, 0)[:, :, -1] for image in images]

    len_images = len(images)
    print(f'Plotting {len_images} images')
    sb_img.ImageGrid(images, cmap='Greys_r', col_wrap=5)

def plot_history_array(history_indices, history, xlabel='Batches', ylabel='Loss'):
    plt.plot(history_indices, history)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
