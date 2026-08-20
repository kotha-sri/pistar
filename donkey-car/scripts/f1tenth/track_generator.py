"""Procedural random track generator for F1TENTH.

Generates closed racing tracks with:
  - Random control points in polar coordinates
  - Cubic spline interpolation for smooth curves
  - Variable track width
  - Occupancy map (PNG + YAML) for simulator collision detection
  - Raceline and centerline CSVs in F1TENTH format
"""

import os
import numpy as np
from scipy.interpolate import CubicSpline
from PIL import Image


def _random_control_points(n_points, r_mean, r_std, rng):
    """Generate random control points around a circle."""
    angles = np.sort(rng.uniform(0, 2 * np.pi, n_points))
    radii = rng.normal(r_mean, r_std, n_points)
    radii = np.clip(radii, r_mean * 0.4, r_mean * 1.6)
    return angles, radii


def _smooth_track(angles, radii, n_interp=500):
    """Interpolate control points into a smooth closed curve."""
    # Close the loop by appending first point at angle + 2*pi
    angles_closed = np.concatenate([angles, [angles[0] + 2 * np.pi]])
    radii_closed = np.concatenate([radii, [radii[0]]])

    cs = CubicSpline(angles_closed, radii_closed, bc_type="periodic")
    t = np.linspace(0, 2 * np.pi, n_interp, endpoint=False)
    r = cs(t)

    x = r * np.cos(t)
    y = r * np.sin(t)
    return x, y


def _arc_length_param(x, y):
    """Compute arc-length parameterization."""
    dx = np.diff(x, append=x[0])
    dy = np.diff(y, append=y[0])
    ds = np.sqrt(dx**2 + dy**2)
    s = np.concatenate([[0], np.cumsum(ds[:-1])])
    return s, ds


def _compute_heading_curvature(x, y):
    """Compute heading (psi) and curvature (kappa) from x,y."""
    dx = np.gradient(x, edge_order=2)
    dy = np.gradient(y, edge_order=2)
    ddx = np.gradient(dx, edge_order=2)
    ddy = np.gradient(dy, edge_order=2)

    psi = np.arctan2(dy, dx)
    speed_sq = dx**2 + dy**2
    kappa = (dx * ddy - dy * ddx) / (speed_sq**1.5 + 1e-10)
    return psi, kappa


def _compute_velocities(kappa, v_max=8.0, mu=0.5, g=9.81):
    """Compute velocity profile from curvature (simplified friction limit)."""
    v_lat_max = np.sqrt(mu * g / (np.abs(kappa) + 1e-6))
    v = np.minimum(v_max, v_lat_max)
    v = np.clip(v, 1.0, v_max)
    # Smooth velocity profile
    from scipy.ndimage import uniform_filter1d
    v = uniform_filter1d(v, size=20, mode="wrap")
    return v


def _render_occupancy_map(x, y, width_r, width_l, psi, resolution=0.05,
                          max_pixels=4_000_000):
    """Render track as a filled polygon occupancy grid map."""
    normal_x = -np.sin(psi)
    normal_y = np.cos(psi)

    left_x = x + width_l * normal_x
    left_y = y + width_l * normal_y
    right_x = x - width_r * normal_x
    right_y = y - width_r * normal_y

    all_x = np.concatenate([left_x, right_x, x])
    all_y = np.concatenate([left_y, right_y, y])
    margin = 5.0
    x_min, x_max = all_x.min() - margin, all_x.max() + margin
    y_min, y_max = all_y.min() - margin, all_y.max() + margin

    # Adapt resolution to keep map size reasonable
    w_est = (x_max - x_min) / resolution
    h_est = (y_max - y_min) / resolution
    if w_est * h_est > max_pixels:
        scale = np.sqrt(w_est * h_est / max_pixels)
        resolution = resolution * scale

    w = int((x_max - x_min) / resolution) + 1
    h = int((y_max - y_min) / resolution) + 1

    img = np.zeros((h, w), dtype=np.uint8)

    # Build closed polygon: left boundary forward, right boundary backward
    poly_x = np.concatenate([left_x, right_x[::-1]])
    poly_y = np.concatenate([left_y, right_y[::-1]])

    # Convert to pixel coordinates
    poly_px = ((poly_x - x_min) / resolution).astype(np.int32)
    poly_py = (h - 1 - ((poly_y - y_min) / resolution)).astype(np.int32)

    # Fill polygon using scanline approach
    from PIL import ImageDraw
    pil_img = Image.fromarray(img)
    draw = ImageDraw.Draw(pil_img)
    polygon_points = list(zip(poly_px.tolist(), poly_py.tolist()))
    draw.polygon(polygon_points, fill=254)
    img = np.array(pil_img)

    origin = [x_min, y_min, 0.0]
    return img, resolution, origin


def generate_track(
    seed=None,
    n_control_points=None,
    r_mean=None,
    r_std=None,
    width_mean=None,
    width_std=None,
    n_interp=800,
):
    """Generate a random closed track.

    Returns:
        dict with keys: raceline, centerline, map_img, map_resolution,
        map_origin, name, seed
    """
    rng = np.random.default_rng(seed)

    if n_control_points is None:
        n_control_points = rng.integers(6, 16)
    if r_mean is None:
        r_mean = rng.uniform(8.0, 25.0)
    if r_std is None:
        r_std = r_mean * rng.uniform(0.1, 0.4)
    if width_mean is None:
        width_mean = rng.uniform(0.8, 1.5)
    if width_std is None:
        width_std = width_mean * 0.15

    angles, radii = _random_control_points(n_control_points, r_mean, r_std, rng)
    x, y = _smooth_track(angles, radii, n_interp)
    s, ds = _arc_length_param(x, y)
    psi, kappa = _compute_heading_curvature(x, y)
    vx = _compute_velocities(kappa)
    ax = np.gradient(vx)

    # Raceline: s, x, y, psi, kappa, vx, ax
    raceline = np.column_stack([s, x, y, psi, kappa, vx, ax])

    # Centerline with variable width
    n = len(x)
    width_r = rng.normal(width_mean, width_std, n)
    width_l = rng.normal(width_mean, width_std, n)
    width_r = np.clip(width_r, 0.5, 2.5)
    width_l = np.clip(width_l, 0.5, 2.5)
    from scipy.ndimage import uniform_filter1d
    width_r = uniform_filter1d(width_r, size=30, mode="wrap")
    width_l = uniform_filter1d(width_l, size=30, mode="wrap")

    # Centerline: x, y, w_right, w_left
    centerline = np.column_stack([x, y, width_r, width_l])

    # Render occupancy map
    map_img, map_res, map_origin = _render_occupancy_map(
        x, y, width_r, width_l, psi, resolution=0.05
    )

    name = f"random_{seed}" if seed is not None else f"random_{rng.integers(0, 100000)}"
    return {
        "raceline": raceline,
        "centerline": centerline,
        "map_img": map_img,
        "map_resolution": map_res,
        "map_origin": map_origin,
        "name": name,
        "seed": seed,
        "total_length": s[-1] + ds[-1],
    }


def save_track(track_data, output_dir):
    """Save generated track to disk in F1TENTH format."""
    name = track_data["name"]
    track_dir = os.path.join(output_dir, name)
    os.makedirs(track_dir, exist_ok=True)

    # Raceline CSV
    raceline_path = os.path.join(track_dir, f"{name}_raceline.csv")
    header = "# generated\n# procedural\n# s_m; x_m; y_m; psi_rad; kappa_radpm; vx_mps; ax_mps2"
    np.savetxt(raceline_path, track_data["raceline"], delimiter=";", header="",
               comments="", fmt="%.7f")
    with open(raceline_path, "r") as f:
        content = f.read()
    with open(raceline_path, "w") as f:
        f.write(header + "\n" + content)

    # Centerline CSV
    centerline_path = os.path.join(track_dir, f"{name}_centerline.csv")
    cl_header = "# x_m, y_m, w_tr_right_m, w_tr_left_m"
    np.savetxt(centerline_path, track_data["centerline"], delimiter=",",
               header="", comments="", fmt="%.7f")
    with open(centerline_path, "r") as f:
        content = f.read()
    with open(centerline_path, "w") as f:
        f.write(cl_header + "\n" + content)

    # Map PNG
    map_path = os.path.join(track_dir, f"{name}_map.png")
    Image.fromarray(track_data["map_img"]).save(map_path)

    # Map YAML
    yaml_path = os.path.join(track_dir, f"{name}_map.yaml")
    origin = track_data["map_origin"]
    with open(yaml_path, "w") as f:
        f.write(f"image: {name}_map.png\n")
        f.write(f"resolution: {track_data['map_resolution']}\n")
        f.write(f"origin: [{origin[0]},{origin[1]}, {origin[2]}]\n")
        f.write("negate: 0\n")
        f.write("occupied_thresh: 0.45\n")
        f.write("free_thresh: 0.196\n")

    return track_dir


def generate_and_save(seed, output_dir):
    """Generate a random track and save it. Returns track directory path."""
    track_data = generate_track(seed=seed)
    return save_track(track_data, output_dir)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate random F1TENTH tracks")
    parser.add_argument("--n-tracks", type=int, default=10)
    parser.add_argument("--output-dir", default="./generated_tracks")
    parser.add_argument("--seed-start", type=int, default=0)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    for i in range(args.n_tracks):
        seed = args.seed_start + i
        track_data = generate_track(seed=seed)
        path = save_track(track_data, args.output_dir)
        print(f"Track {seed}: length={track_data['total_length']:.1f}m, saved to {path}")
