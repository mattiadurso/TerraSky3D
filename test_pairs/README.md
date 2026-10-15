# TerraSky3D — test pairs for pairwise matching / relative pose

43,720 image pairs over 12 test scenes, with ground-truth intrinsics and relative pose.

| Scene | Pairs |
| :--- | ---: |
| graz_church | 6,184 |
| graz_townhall | 4,889 |
| graz_university | 6,652 |
| munich_frauenkirche | 947 |
| munich_marienplatz | 3,849 |
| munich_theatine_church | 1,551 |
| salzburg_andrakirche | 1,972 |
| salzburg_recthe_altstadt | 464 |
| udine_devils_bridge | 6,961 |
| udine_fagagna_church | 2,978 |
| udine_villalta_castle | 6,143 |
| vienna_state_opera | 1,130 |

## Files

- `scenes/<scene>.txt`: the pairs of one scene.
- `pairs_all.txt`: all scenes concatenated.
- `pairs_all.json`: the same pairs grouped by scene, plus image size and radial distortion per camera.

## TXT format

One pair per line, the two image paths followed by 30 numbers:

```
img0 img1  K0(9)  K1(9)  R(9)  t(3)
```

- Image paths are relative to the test data root: `<scene>/frames/<camera>/<image>.jpg`.
- `K0`, `K1`: 3x3 pinhole intrinsics, row-major, in pixels at the stored (4K) image resolution.
- `R`, `t`: relative pose from camera 0 to camera 1, `x1 = R @ x0 + t`, with COLMAP world-to-camera poses: `R = R1 @ R0.T`, `t = t1 - R @ t0`. `t` is normalised to unit length, so it gives only the translation direction.

## JSON format

```json
{
  "scenes": {
    "<scene>": {
      "num_pairs": 0,
      "pairs": [
        {"image0": "...", "image1": "...",
         "K0": [[...]], "K1": [[...]], "R_0to1": [[...]], "t_0to1": [...],
         "cam0": {"fx": 0, "fy": 0, "cx": 0, "cy": 0, "width": 0, "height": 0, "k1": 0},
         "cam1": {"...": "..."}}
      ]
    }
  }
}
```

The cameras are COLMAP `SIMPLE_RADIAL` models and the images are **not** undistorted. `k1` is the radial term. It is small, and the reference benchmark ignores it and uses only `K`.

## Evaluation

The standard protocol, as in MegaDepth-1500: estimate the essential matrix from the matches with `K0` and `K1`, and report the AUC of the pose error at 5°, 10° and 20°. The pose error is the larger of the rotation and translation angular errors.
