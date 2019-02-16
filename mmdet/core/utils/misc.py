import mmcv
import numpy as np
from functools import partial
from six.moves import map, zip
from subprocess import Popen, PIPE


def tensor2imgs(tensor, mean=(0, 0, 0), std=(1, 1, 1), to_rgb=True):
    num_imgs = tensor.size(0)
    mean = np.array(mean, dtype=np.float32)
    std = np.array(std, dtype=np.float32)
    imgs = []
    for img_id in range(num_imgs):
        img = tensor[img_id, ...].cpu().numpy().transpose(1, 2, 0)
        img = mmcv.imdenormalize(
            img, mean, std, to_bgr=to_rgb).astype(np.uint8)
        imgs.append(np.ascontiguousarray(img))
    return imgs


def multi_apply(func, *args, **kwargs):
    pfunc = partial(func, **kwargs) if kwargs else func
    map_results = map(pfunc, *args)
    return tuple(map(list, zip(*map_results)))


def unmap(data, count, inds, fill=0):
    """ Unmap a subset of item (data) back to the original set of items (of
    size count) """
    if data.dim() == 1:
        ret = data.new_full((count, ), fill)
        ret[inds] = data
    else:
        new_size = (count, ) + data.size()[1:]
        ret = data.new_full(new_size, fill)
        ret[inds, :] = data
    return ret


def select_gpus(N, max_utilization=.5, max_memory_usage=.5):
    cmd = ["nvidia-smi",
           "--query-gpu=index,utilization.gpu,memory.total,memory.used",
           "--format=csv,noheader,nounits"]
    p = Popen(cmd, stdout=PIPE)
    output = p.stdout.read().decode('UTF-8')
    gpus = [[int(x) for x in x.split(',')] for x in output.splitlines()]
    gpu_ids = []
    for (index, utilization, total, used) in gpus:
        if utilization / 100.0 < max_utilization:
            if used * 1.0 / total < max_memory_usage:
                gpu_ids.append(index)
    selected = np.random.choice(gpu_ids, size=(N,), replace=False)
    return list(selected)
