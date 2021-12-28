import matplotlib.pyplot as plt
import numpy as np
import csv

def set_axes_equal(ax: plt.Axes):
    """Set 3D plot axes to equal scale.

    Make axes of 3D plot have equal scale so that spheres appear as
    spheres and cubes as cubes.  Required since `ax.axis('equal')`
    and `ax.set_aspect('equal')` don't work on 3D.
    """
    limits = np.array([
        ax.get_xlim3d(),
        ax.get_ylim3d(),
        ax.get_zlim3d(),
    ])
    origin = np.mean(limits, axis=1)
    radius = 0.5 * np.max(np.abs(limits[:, 1] - limits[:, 0]))
    _set_axes_radius(ax, origin, radius)

def _set_axes_radius(ax, origin, radius):
    x, y, z = origin
    ax.set_xlim3d([x - radius, x + radius])
    ax.set_ylim3d([y - radius, y + radius])
    ax.set_zlim3d([z - radius, z + radius])

side = "left"
applyTransform = True
plotLabels = False

paths = [f"points_T265_{side}.csv", f"points_Eye_{side}.csv"]
#paths = [f"points_T265Camera.csv", f"points_LeapMotion.csv"]
#paths = [f"points_LeapMotion.csv"]

r = [[ 0.54797501,-0.8209729 ,-0.16039606],
 [-0.70934996,-0.55768657, 0.43105491],
 [-0.44333512,-0.12243038,-0.88795539]]
t = [15.10298607,-2.1485197 ,52.115229  ]

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.set_box_aspect([1,1,1])

for i, path in enumerate(paths):
    with open(path, newline='') as f:
        reader = csv.reader(f, quoting=csv.QUOTE_NONNUMERIC)
        data = np.array(list(reader))
        if i == 0 and applyTransform is True and r is not None and t is not None:
            for j, row in enumerate(data):
                data[j] = np.dot(r, row) + t
        xs, ys, zs = data.T
        ax.scatter(xs, ys, zs, s=2, label=path)
        if plotLabels:
            for j in range(len(xs)):
                ax.text(xs[j], ys[j], zs[j], j)

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
ax.legend()
set_axes_equal(ax)
plt.show()
