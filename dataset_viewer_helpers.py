# visual test: can I reproject the points correctly?
import torch
from mylib.projections import reproject_2D_2D


def create_grid(image, permute=False, sampling_factor=10, border=30):
    """
    Function to create a grid of the same size as the image.
    Args:
        image: image of shape BxCxHxW or CxHxW
        permute: if True, the grid is permuted
    Returns:
        grid: grid of the same size as the image HxWx2
    """
    image = image[None] if image.dim() == 3 else image
    H, W = image.shape[-2:]

    grid_y, grid_x = torch.meshgrid(
        torch.arange(border, H - border, sampling_factor),
        torch.arange(border, W - border, sampling_factor),
        indexing="ij",
    )
    grid = torch.stack((grid_x, grid_y), dim=-1).view(-1, 2).float()

    grid = grid[torch.randperm(grid.shape[0])] if permute else grid

    return grid


def dist(p0, p1):
    """
    Euclidean distance between two points
    Args:
        p0: point 0 (N,2)
        p1: point 1 (N,2)
    Returns:
        dist: distance between the points (N,)
    """
    return torch.sqrt(((p0 - p1) ** 2).sum(dim=-1))


def compute_121_reprojection(
    data,
    img0,
    img1,
    verbose=True,
    reprojection_error=3.0,
    border=30,
    sampling_factor=10,
):
    # create a grid of points in img 0
    kpts0 = create_grid(img0, sampling_factor=sampling_factor, border=border)[None]
    # starting from depth valid locations, in nan is invalid in any case
    # kpts0 = torch.nonzero(~torch.isnan(data['depth0'][0]))[None].float() # why not working?
    tot = kpts0.numel()

    # project the points to img1
    kpts1 = reproject_2D_2D(
        xy0=kpts0,
        depthmap0=data["depth0"],
        P0=data["P0"],
        P1=data["P1"],
        K0=data["K0"],
        K1=data["K1"],
        img1_shape=(img1.shape[-2], img1.shape[-1]),
    )

    # back project the points to img0
    kpts0_back = reproject_2D_2D(
        xy0=kpts1,
        depthmap0=data["depth1"],
        P0=data["P1"],
        P1=data["P0"],
        K0=data["K1"],
        K1=data["K0"],
        img1_shape=(img0.shape[-2], img0.shape[-1]),
    )

    if verbose:
        print(kpts0.shape, kpts1.shape, kpts0_back.shape, "projected")

    # detect nans and remove if any, no need for kpts0
    nan_mask = torch.logical_and(
        torch.isnan(kpts1).any(dim=-1), torch.isnan(kpts0_back)[0].any(dim=-1)
    )
    kpts0 = kpts0[~nan_mask]
    kpts1 = kpts1[~nan_mask]
    kpts0_back = kpts0_back[~nan_mask]

    if verbose:
        print(kpts0.shape, kpts1.shape, "removed nan")

    # check if back projections is close enough to the original points
    distances = dist(kpts0, kpts0_back)
    mask = distances < reprojection_error
    kpts0 = kpts0[mask]
    kpts1 = kpts1[mask]

    # check projection to be within border margin for kpt1
    mask_x = torch.logical_and(
        kpts1[:, 0] > border, kpts1[:, 0] < img1.shape[-1] - border
    )
    mask_y = torch.logical_and(
        kpts1[:, 1] > border, kpts1[:, 1] < img1.shape[-2] - border
    )
    mask = torch.logical_and(mask_x, mask_y)
    kpts0 = kpts0[mask]
    kpts1 = kpts1[mask]

    return kpts0, kpts1, tot, distances
