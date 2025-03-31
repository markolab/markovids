import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from matplotlib.animation import FFMpegWriter
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection


def visualize_xyz_trajectories_to_mp4(
    xyz,  # (T, N, 3)
    output_path,
    skeleton_edges=None,
    fps=100,
    elev=15,
    azim=85,
    point_size=30,
    figsize=(8, 6),
    zlim=(-10, 125),
    xlim=(-200, 300),
    ylim=(-300, 200),
    show_labels=False,
    tick_spacing_floor=100,
    tick_spacing_height=25,
    color_map="tab10",
    trail_length=10,
):
    T, N, _ = xyz.shape
    cmap = plt.get_cmap(color_map)
    keypoint_colors = [cmap(i % cmap.N) for i in range(N)]

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")
    ax.set_autoscale_on(False)
    writer = FFMpegWriter(fps=fps)

    # Setup 3D scatter objects
    scatters = [
        ax.scatter([0], [0], [0], s=point_size, color=keypoint_colors[i])
        for i in range(N)
    ]

    # Text labels
    texts = []
    if show_labels:
        texts = [ax.text(0, 0, 0, str(i), fontsize=6) for i in range(N)]

    # Skeleton lines
    skeleton_lines = []
    if skeleton_edges:
        for i, j in skeleton_edges:
            (line,) = ax.plot([], [], [], color="gray", lw=1)
            skeleton_lines.append((line, i, j))

    # Start video writer
    if trail_length > 0:
        trail_manager = KeypointTrails3D(ax, N, trail_length, colors=keypoint_colors)
    add_height_grid(ax, xlim, ylim, zlim, tick_spacing=tick_spacing_height)
    add_floor_grid(ax, xlim, ylim, zlim, tick_spacing=tick_spacing_floor)

      # Axes, view, and limits
    ax.view_init(elev=elev, azim=azim)

    if xlim:
        ax.set(xlim=xlim, xticks=[], xlabel="")
    if ylim:
        ax.set(ylim=ylim, yticks=[], ylabel="")
    if zlim:
        ax.set(zlim=zlim, zticks=[], zlabel="")
    
    with writer.saving(fig, output_path, dpi=200):
        for t in tqdm(range(T)):
            frame_xyz = xyz[t]

            # Update scatter positions
            for i, scatter in enumerate(scatters):
                if np.all(np.isfinite(frame_xyz[i])):
                    scatter._offsets3d = (
                        [frame_xyz[i, 0]],
                        [frame_xyz[i, 1]],
                        [frame_xyz[i, 2]],
                    )
                else:
                    scatter._offsets3d = ([np.nan], [np.nan], [np.nan])

            # Update labels
            if show_labels:
                for i, txt in enumerate(texts):
                    if np.all(np.isfinite(frame_xyz[i])):
                        txt.set_position((frame_xyz[i, 0], frame_xyz[i, 1]))
                        txt.set_3d_properties(frame_xyz[i, 2])
                    else:
                        txt.set_position((0, 0))
                        txt.set_3d_properties(0)

            # Update skeletons
            if skeleton_edges:
                for line, i, j in skeleton_lines:
                    if np.all(np.isfinite([frame_xyz[i], frame_xyz[j]])):
                        x = [frame_xyz[i, 0], frame_xyz[j, 0]]
                        y = [frame_xyz[i, 1], frame_xyz[j, 1]]
                        z = [frame_xyz[i, 2], frame_xyz[j, 2]]
                        line.set_data(x, y)
                        line.set_3d_properties(z)
                    else:
                        line.set_data([], [])
                        line.set_3d_properties([])

          
            ax.set_title(f"Frame {t+1}/{T}")
    
            # ax.set_autoscale_on(False)
            if trail_length>0:
                trail_manager.update(frame_xyz)

            writer.grab_frame()
            # ax.cla()  # clear only at end of frame to avoid redrawing everything
            

class KeypointTrails3D:
    def __init__(self, ax, N, trail_length=10, colors=None, alpha_max=0.8):
        self.N = N
        self.trail_length = trail_length
        self.buffers = [[] for _ in range(N)]
        self.collections = []
        self.alpha_max = alpha_max

        for i in range(N):
            lc = Line3DCollection([], linewidths=1)
            lc.set_color(colors[i] if colors else 'black')
            lc.set_segments([[(0, 0, 0), (0, 0, 0)]])  # initialize to prevent auto_scale_xyz error
            ax.add_collection3d(lc)
            self.collections.append(lc)

    def update(self, frame_xyz):
        for i in range(self.N):
            if np.all(np.isfinite(frame_xyz[i])):
                self.buffers[i].append(frame_xyz[i])
                if len(self.buffers[i]) > self.trail_length:
                    self.buffers[i] = self.buffers[i][-self.trail_length:]
            else:
                self.buffers[i] = []

            trail = np.array(self.buffers[i])
            if len(trail) >= 2:
                segments = np.stack([trail[:-1], trail[1:]], axis=1)
                num_segs = len(segments)
                alphas = np.linspace(0.1, self.alpha_max, num_segs)
                base_color = self.collections[i].get_color()[0]
                colors = [(*base_color[:3], a) for a in alphas]
                self.collections[i].set_segments(segments)
                self.collections[i].set_color(colors)
            else:
                self.collections[i].set_segments([])
       

def add_height_grid(ax, xlim, ylim, zlim, tick_spacing=50):
    zmin, zmax = zlim
    xmin, xmax = xlim
    ymin, ymax = ylim

    z_ticks = np.arange(np.ceil(zmin / tick_spacing) * tick_spacing,
                        np.floor(zmax / tick_spacing) * tick_spacing + 1, tick_spacing)

    grid_lines = []
    for z in z_ticks:
        grid_lines.append([(xmin, ymin, z), (xmax, ymin, z)])
        grid_lines.append([(xmin, ymax, z), (xmax, ymax, z)])
        grid_lines.append([(xmin, ymin, z), (xmin, ymax, z)])
        grid_lines.append([(xmax, ymin, z), (xmax, ymax, z)])

    line_collection = Line3DCollection(grid_lines, colors='lightblue', linestyles=':', linewidths=1, alpha=0.5)
    ax.add_collection3d(line_collection)

    text_offset = 1e-3
    for z in z_ticks:
        ax.text(xmin, ymin, z + text_offset, f"{int(z)}", color='lightblue', fontsize=7, ha='right', va='bottom')


def add_floor_grid(ax, xlim, ylim, zlim, tick_spacing=50):
    xmin, xmax = xlim
    ymin, ymax = ylim
    zmin, _ = zlim
    floor_z = zmin

    floor_coords = [
        (xmin, ymin, floor_z), (xmax, ymin, floor_z),
        (xmax, ymax, floor_z), (xmin, ymax, floor_z)
    ]
    floor_plane = Poly3DCollection([floor_coords], facecolor='gray', edgecolor='none', alpha=0.2)
    ax.add_collection3d(floor_plane)

    x_ticks = np.arange(np.ceil(xmin / tick_spacing) * tick_spacing,
                        np.floor(xmax / tick_spacing) * tick_spacing + 1, tick_spacing)
    y_ticks = np.arange(np.ceil(ymin / tick_spacing) * tick_spacing,
                        np.floor(ymax / tick_spacing) * tick_spacing + 1, tick_spacing)

    grid_lines = []
    for x in x_ticks:
        grid_lines.append([(x, ymin, floor_z), (x, ymax, floor_z)])
    for y in y_ticks:
        grid_lines.append([(xmin, y, floor_z), (xmax, y, floor_z)])
    line_collection = Line3DCollection(grid_lines, colors='gray', linestyles=':', linewidths=1, alpha=0.5)
    ax.add_collection3d(line_collection)

    text_elev = 1e-3
    for x in x_ticks:
        ax.text(x, ymin - 10, floor_z + text_elev, f"{int(x)}", color='gray', fontsize=8, ha='center', va='bottom')
    for y in y_ticks:
        ax.text(xmin - 10, y, floor_z + text_elev, f"{int(y)}", color='gray', fontsize=8, ha='left', va='center')

