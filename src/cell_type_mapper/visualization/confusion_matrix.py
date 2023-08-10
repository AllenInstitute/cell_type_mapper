from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np


def plot_confusion_matrix(
        figure,
        axis,
        true_labels,
        experimental_labels,
        label_order,
        normalize_by='truth',
        fontsize=20,
        title=None,
        is_log=False,
        munge_ints=True,
        label_x_axis=True,
        label_y_axis=True):

    img = np.zeros((len(label_order), len(label_order)), dtype=int)
    label_to_idx = {
        l: ii for ii, l in enumerate(label_order)}

    for truth, experiment in zip(true_labels, experimental_labels):
        true_idx = label_to_idx[truth]
        experiment_idx = label_to_idx[experiment]
        img[true_idx, experiment_idx] += 1

    s0 = img.sum()
    img, thinned_labels = thin_img(img, label_list=label_order)
    assert img.sum() == s0

    img = np.ma.masked_array(
        img, mask=(img == 0))

    img = img.astype(float)

    if normalize_by == 'truth':
        for ii in range(img.shape[0]):
            denom = img[ii, :].sum()
            img[ii, :] /= max(1, denom)
    elif normalize_by == 'experiment':
        for ii in range(img.shape[1]):
            denom = img[:, ii].sum()
            img[:, ii] /= max(1, denom)
    else:
        raise RuntimeError(
            f"normalize_by {normalize_by} makes no sense")

    if is_log:
        cax_title = 'log10(normalized count)'
        with np.errstate(divide='ignore'):
            valid = (img > 0.0)
            min_val = np.log10(np.min(img[valid]))
            img = np.where(
                img > 0.0,
                np.log10(img),
                min_val-2)
    else:
        cax_title = 'normalized count'

    display_img = axis.imshow(img, cmap='cool')
    divider = make_axes_locatable(axis)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    figure.colorbar(
        display_img,
        ax=axis,
        cax=cax,
        label=cax_title)

    for s in ('top', 'right', 'left', 'bottom'):
        axis.spines[s].set_visible(False)

    if label_x_axis:
        axis.set_xlabel('mapped label', fontsize=fontsize)
    if label_y_axis:
        axis.set_ylabel('true label', fontsize=fontsize)

    if title is not None:
        axis.set_title(title, fontsize=fontsize)

    tick_values = [ii for ii in range(len(thinned_labels))]
    axis.set_xticks(tick_values)
    axis.set_xticklabels(thinned_labels, fontsize=15, rotation='vertical')
    axis.set_yticks(tick_values)
    axis.set_yticklabels(thinned_labels, fontsize=15, rotation='horizontal')


def thin_img(img, label_list):
    n_el = img.shape[0]
    to_keep = []
    for ii in range(n_el):
        if img[ii, :].sum() > 0 or img[:, ii].sum() > 0:
            to_keep.append(ii)
    to_keep = np.array(to_keep)
    img = img[to_keep, :]
    img = img[:, to_keep]
    return img, np.array(label_list)[to_keep]
