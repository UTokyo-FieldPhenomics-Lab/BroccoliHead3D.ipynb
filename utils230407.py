import os
from copy import copy

from tqdm import tqdm

import numpy as np
import pandas as pd
import open3d as o3d

from scipy import stats
from scipy.spatial import ConvexHull

from sklearn.cluster import KMeans
from skimage import filters
from skimage.measure import regionprops
from skimage.color import rgb2hsv

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from mpl_toolkits.mplot3d import Axes3D
import mpl_toolkits.mplot3d.art3d as art3d
from mpl_toolkits import mplot3d

from geometry.min_bounding_rect import min_bounding_rect
from geometry.fit_ellipse import fit_ellipse

o3d_xyz_coord = o3d.geometry.LineSet()

o3d_xyz_coord.points = o3d.utility.Vector3dVector(np.array([[0,0,0],[0.1,0,0],[0,0.1,0], [0,0,0.1]]))
o3d_xyz_coord.lines = o3d.utility.Vector2iVector([[0,1],[0,2],[0,3]])
o3d_xyz_coord.colors = o3d.utility.Vector3dVector(np.array([[1,0,0],[0,1,0],[0,0,1]]))

def mesh2pcd(mesh):
    mesh_pcd = o3d.geometry.PointCloud()
    mesh_pcd.points = mesh.vertices
    mesh_pcd.colors = mesh.vertex_colors
    mesh_pcd.normals = mesh.vertex_normals
    
    return mesh_pcd

def points_on_vector(points, vector):
    """
    把点投影到给定向量上
    输入：points: nx3的三维点
         vector: 1x3的向量
    输出：投影后的点(在空间中的位置)
    """
    v_norm = np.sqrt(sum(vector**2))
    
    return (np.dot(points, vector.reshape(3,1))/v_norm**2)*vector

def points_on_plane(points, plane, mode='vector'):
    """
    mode is vector: plane is (3,) array of the normal vector
                  (nx, ny, nz)
    mode is abcd: plane is (4,) array of the plane expression 
                  (A, B, C ,D) -> Ax + By + Cz + D =0
    """
    if mode == 'vector':
        plane_normal = plane / np.linalg.norm(plane)
        projection = points - np.dot(points, plane_normal)[:, None] * plane_normal

        return projection
    else:
        # ref: https://blog.csdn.net/soaryy/article/details/82884691
        # Plane_params = (A, B, C ,D) -> Ax + By + Cz + D =0
        A, B, C, D = plane
        demomin = A**2 + B**2 + C**2
        x0, y0, z0 = points[:,0], points[:,1], points[:,2]
    
        xp = ((B**2+C**2)*x0 - A*(B*y0+C*z0+D)) / demomin
        yp = ((A**2+C**2)*y0 - B*(A*x0+C*z0+D)) / demomin
        zp = ((A**2+B**2)*z0 - C*(A*x0+B*y0+D)) / demomin
    
        return np.vstack([xp, yp, zp]).T

def segment_crown(head_pcd, denoise_count=20, denoise_radius=0.01, show=False):
    # 计算定向边界框
    obb = head_pcd.get_oriented_bounding_box()
    # find the maximum edge and its vector:
    main_idx = np.argmax(obb.extent)

    # use Kmeans to classify points to 2 class by color
    rgb = np.asarray(head_pcd.colors)
    kmeans = KMeans(n_clusters=2, random_state=0, n_init=10)
    kmeans.fit(rgb)

    rgb_class0 = rgb[kmeans.labels_ == 0, :]
    rgb_class1 = rgb[kmeans.labels_ == 1, :]

    hsv_class0 = rgb2hsv(rgb_class0.mean(axis=0))
    hsv_class1 = rgb2hsv(rgb_class1.mean(axis=0))

    if hsv_class0[2] < hsv_class1[2]:
        crown_bool_idx = kmeans.labels_ == 0
    else:
        crown_bool_idx = kmeans.labels_ == 1

    crown_int_idx = np.where(crown_bool_idx)[0]

    crown_pcd = head_pcd.select_by_index(crown_int_idx)
    # crown_pcd_denoised, keeped_int_idx = crown_pcd.remove_radius_outlier(denoise_count, denoise_radius)
    # crown_pcd_denoised, keeped_int_idx = crown_pcd.remove_statistical_outlier(denoise_count, denoise_radius)
    crown_pcd_denoised, keeped_int_idx = remove_dbscan_outlier(crown_pcd, denoise_count, denoise_radius)
    crown_denoised_int_idx = crown_int_idx[keeped_int_idx]

    ## create a results pcd, 
    labeled_pcd = o3d.geometry.PointCloud()

    ### black for stem
    label_rgb = np.zeros((len(head_pcd.points), 3))
    ### red for denoised
    label_rgb[crown_bool_idx, :] = np.array([1,0,0])
    ### green for crown
    label_rgb[crown_denoised_int_idx,:] = np.array([0,1,0])

    labeled_pcd.points = o3d.utility.Vector3dVector(head_pcd.points)
    labeled_pcd.colors = o3d.utility.Vector3dVector(label_rgb)

    #################
    ## draw figures
    #################
    if show:
        # 把点云投影到最长轴上
        # pcd_on_z -> (x, y, z) 3D
        pcd_on_z = points_on_vector(np.asarray(head_pcd.points), obb.R[:,main_idx])
        # project 3D line to 1D
        pcd_on_z_unit = (pcd_on_z / obb.R[:,main_idx])[:,0]

        ## the z axis points of each class
        z0 = pcd_on_z_unit[kmeans.labels_ == 0]
        z1 = pcd_on_z_unit[kmeans.labels_ == 1]
        z0_mean = z0.mean()
        z1_mean = z1.mean()

        # find out which is the crown class
        ## calculate the direction
        pmax = pcd_on_z_unit.max()
        pmin = pcd_on_z_unit.min()
        pmid = (pmax + pmin) / 2
        pmean = pcd_on_z_unit.mean()

        kde0 = stats.gaussian_kde(z0)
        xx0 = np.linspace(pmin, pmax, 1000)
        yy0 = kde0(xx0)

        kde1 = stats.gaussian_kde(z1)
        xx1 = np.linspace(pmin, pmax, 1000)
        yy1 = kde1(xx1)

        fig, ax = plt.subplots(1,1)
        ax.plot(xx0, yy0,color='b', label='z0')
        ax.plot(xx1, yy1,color='g', label='z1')

        ax.axvline(x=pmid, color='r', label="middle")

        ax.axvline(x=z0_mean, color='b', label="z0mean")
        ax.axvline(x=z1_mean, color='g',label="z1mean")

        plt.legend()
        plt.show()

        plt.clf()
        plt.cla()
        plt.close(fig)

    return crown_pcd_denoised, labeled_pcd, crown_denoised_int_idx

#====================

def remove_dbscan_outlier(pcd, min_points, eps, show_log=False):
    label = pcd.cluster_dbscan(eps=eps, min_points=min_points, print_progress=False)
    db_labels, db_count = np.unique(label, return_counts=True)

    if len(db_count) <= 2:  # already splited by [-1 ,0]
        broccoli_crown_cluster_idx = [0]
    else:
        # calculate the cluster size as another reference
        characters = np.empty((len(db_count), 2))
        for i, (l, c) in enumerate(zip(db_labels, db_count)):
            idx_on_label = np.where(label==l)[0]
            pcd_on_label = pcd.select_by_index(idx_on_label)
            volume = _calculate_xyz_volume(pcd_on_label)

            characters[i-1, 0] = c
            characters[i-1, 1] = volume

        # print("charectors:\n",characters)
        # Kmeans cluster
        km = KMeans(n_clusters=2, n_init=10)
        km.fit(characters)

        class0 = characters[km.labels_ == 0, :]
        class1 = characters[km.labels_ == 1, :]

        # print("class0, class1\n", class0, '\n', class1)
        # find the class label with largest point clouds (broccoli_crowns)
        if class0.mean(axis=0)[0] > class1.mean(axis=0)[0]:
            plant_cluster_idx = np.where(km.labels_ == 0)[0]
        else:
            plant_cluster_idx = np.where(km.labels_ == 1)[0]

        if show_log: print(f"DBSCAN\nlabel - count\n{np.vstack([db_labels, db_count]).T}\nUse {plant_cluster_idx} as plants")
        plant_id = []
        for idx in plant_cluster_idx:
            # print(idx, label, np.where(np.array(label)==idx)[0])
            plant_id += np.where(np.array(label)==idx)[0].tolist()

        # print("plant_id\n", plant_id)

        return pcd.select_by_index(plant_id), plant_id

def _calculate_xyz_volume(pcd):
    pcd_xyz = np.asarray(pcd.points)

    x_len = pcd_xyz[:, 0].max() - pcd_xyz[:, 0].min()
    y_len = pcd_xyz[:, 1].max() - pcd_xyz[:, 1].min()
    z_len = pcd_xyz[:, 2].max() - pcd_xyz[:, 2].min()

    return x_len * y_len * z_len


# ==============================
# updated in 07_find_z_by_shape.ipynb
def pcd2binary_2d(xy, dpi=10):
    # dpi suggest < 20
    # !!!! notice !!!!
    # in numpy image system, Y axis is 0, X axis is 1
    y = xy[:, 0]
    x = xy[:, 1]

    x_length_m = x.max() - x.min()
    y_length_m = y.max() - y.min()
    px_num_per_cm = int(dpi / 2.54)
    width = int(np.ceil(x_length_m * 100 * px_num_per_cm))
    height = int(np.ceil(y_length_m * 100 * px_num_per_cm))
    ref_x = (x - x.min()) / x_length_m * width
    ref_y = (y - y.min()) / y_length_m * height
    ref_pos = np.vstack([ref_x, ref_y]).T.astype(int)
    ref_pos_rm_dup = np.unique(ref_pos, axis=0)

    out_img = np.zeros((width + 1, height + 1))
    out_img[ref_pos_rm_dup[:, 0], ref_pos_rm_dup[:, 1]] = 1
    out_img = out_img.astype(int)

    left_top_corner = (y.min(), x.min())

    return out_img, px_num_per_cm, left_top_corner


def rotate_to_z(crown_pcd, original_pcd, show=False, save_path=None):
    obb = crown_pcd.get_oriented_bounding_box()

    xyz = np.asarray(original_pcd.points)
    rgb = np.asarray(original_pcd.colors)

    if save_path is not None or show: 
        fig, ax = plt.subplots(3,3, figsize=(10,12))

    roundness = []
    direction = []
    for i in range(3):
        project_0 = points_on_plane(xyz, obb.R[:, i])

        # get the projected points on the xy coordiantes
        vx = get_unit_vector(obb.R[:, (i+1)%3])
        vy = get_unit_vector(obb.R[:, (i+2)%3])

        x0 = np.dot(project_0, vx)
        y0 = np.dot(project_0, vy)

        xy = np.vstack([x0, y0]).T

        # the vector itself, find the direction
        vz = points_on_vector(xyz, obb.R[:, i])
        z0 = (vz / obb.R[:,i])[:,0]
        zmin = z0.min()
        zmax = z0.max()
        zmid = (z0.min() + z0.max()) / 2
        zmean = z0.mean()
        kde = stats.gaussian_kde(z0)
        xx = np.linspace(zmin, zmax, 1000)
        yy = kde(xx)

        if zmean > zmid:
            direction.append(1)
        else:
            direction.append(-1)

        # calculate the roundness
        out_img, px_num_per_cm, left_top_corner = pcd2binary_2d(xy)
        regions = regionprops(out_img)
        props = regions[0]
        ratio = 4 * np.pi * props.area / ( props.perimeter**2 )
        roundness.append(ratio)

        if save_path is not None or show: 
            ax[0, i].scatter(np.zeros(z0.shape), z0, color=rgb, marker='.', s=0.1)
            ax[0, i].plot(yy, xx, label='kde')
            ax[0, i].axhline(y=zmid, color='r', label='middle')
            ax[0, i].axhline(y=zmean, color='g', label='mean')
            ax[0, i].set_title(f"Points along OBB axis {i}")
            ax[0, i].legend(loc='lower center')

            ax[1, i].scatter(x0, y0, color=rgb, marker='.', s=0.1)
            ax[1, i].axis('equal')
            ax[1, i].set_title(f"Project on OBB axis {i}")

            ax[2, i].imshow(out_img, cmap='gray')
            ax[2, i].set_title(f"roundness = {round(ratio, 2)}")

    main_idx = np.array(roundness).argmax()

    main_direction = obb.R[:, main_idx] * direction[main_idx]

    R = get_rotation_matrix(get_unit_vector(main_direction), (0,0,1))
    pcd_rotated = copy(original_pcd).rotate(R)

    if save_path is not None or show: 
        # set the main direction to red color
        for i in range(3):
            ax[i, main_idx].tick_params(color='green', labelcolor='green')
            for spine in ax[i, main_idx].spines.values():
                spine.set_edgecolor('red')

        if show: 
            plt.show()
        else: 
            plt.savefig(save_path)

        plt.cla()
        plt.clf()
        plt.close(fig)

    return pcd_rotated, R


def angle2(v1, v2):
    x=np.array(v1)
    y=np.array(v2)

    # 分别计算两个向量的模：
    module_x=np.sqrt(x.dot(x))
    module_y=np.sqrt(y.dot(y))

    # 计算两个向量的点积
    dot_value=x.dot(y)

    # 计算夹角的cos值：
    cos_theta=dot_value/(module_x*module_y)

    # 求得夹角（弧度制）：
    angle_radian=np.arccos(cos_theta)

    # 转换为角度值：
    angle_value=angle_radian*180/np.pi
    return angle_value

def get_unit_vector(v):
    return v / np.linalg.norm(v)

def get_rotation_matrix(a, b):
    v = np.cross(a, b)
    s = np.dot(a, b)
    vx = np.array([[0, -v[2], v[1]],
                   [v[2], 0, -v[0]],
                   [-v[1], v[0], 0]])
    R = np.identity(3) + vx + np.dot(vx, vx) * (1 / (1 + s))

    return R


# ===========================================
def draw_rotated(xyz_o, rgb_o, xyz_r, rgb_r, save_path=None, show=True):
    fig = plt.figure(figsize=(15, 10), dpi=300)
    ax1 = fig.add_subplot(231, projection='3d')
    ax2 = fig.add_subplot(232, projection='3d')
    ax3 = fig.add_subplot(233, projection='3d')
    ax4 = fig.add_subplot(234, projection='3d')
    ax5 = fig.add_subplot(235, projection='3d')
    ax6 = fig.add_subplot(236, projection='3d')

    for ax in [ax1, ax2, ax3]:
        ax.scatter(*xyz_o.T, c=rgb_o, marker='.', alpha=0.5, s=1)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        ax.set_box_aspect((np.ptp(xyz_o[:,0]), np.ptp(xyz_o[:,1]), np.ptp(xyz_o[:,2])))

    for ax in [ax4, ax5, ax6]:
        ax.scatter(*xyz_r.T, c=rgb_r, marker='.', alpha=0.5, s=1)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_box_aspect((np.ptp(xyz_r[:,0]), np.ptp(xyz_r[:,1]), np.ptp(xyz_r[:,2])))

    ax1.view_init(elev=0, azim=45)
    ax2.view_init(elev=0, azim=165)
    ax3.view_init(elev=0, azim=285)
    ax4.view_init(elev=0, azim=45)
    ax5.view_init(elev=0, azim=165)
    ax6.view_init(elev=0, azim=285)

    ax1.set_title('45 degree azim')
    ax2.set_title('165 degree azim')
    ax3.set_title('285 degree azim')

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path)
    if show:
        plt.show()

    plt.cla()
    plt.clf()
    plt.close(fig)


# ========================
#         EasyDCP 
# ========================
def get_convex_hull(pcd, dim='2d'):
    # in scipy, 2D hull.area is perimeter, hull.volume is area
    # https://stackoverflow.com/questions/35664675/in-scipys-convexhull-what-does-area-measure
    #
    # >>> points = np.array([[-1,-1], [1,1], [-1, 1], [1,-1]])
    # >>> hull = ConvexHull(points)
    # ==== 2D ====
    # >>> print(hull.volume)
    # 4.00
    # >>> print(hull.area)
    # 8.00
    # ==== 3D ====
    # >>> points = np.array([[-1,-1, -1], [-1,-1, 1],
    # ...                    [-1, 1, -1], [-1, 1, 1],
    # ...                    [1, -1, -1], [1, -1, 1],
    # ...                    [1,  1, -1], [1,  1, 1]])
    # >>> hull = ConvexHull(points)
    # >>> hull.area
    # 24.0
    # >>> hull.volume
    # 8.0
    pcd_xyz = np.asarray(pcd.points)
    xy = pcd_xyz[:, 0:2]
    if dim == '2d' or dim == '2D':
        hull = ConvexHull(xy)
    elif dim == '3d' or dim == '3D':
        hull = ConvexHull(pcd_xyz)
    else:
        raise KeyError('Only "2d" and "3d" or "2D" and "3D" are acceptable for dim parameters')
    hull_volume = hull.volume
    hull_xy = xy[hull.vertices, :]
    return hull_xy, hull_volume, hull

def get_concave_hull_3d(pcd, alpha_st=0.005, alpha_step=0.005):
    alpha_value = alpha_st
    alpha_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha_value)
    while not alpha_mesh.is_watertight():
        alpha_value += alpha_step
        alpha_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha_value)

    return alpha_mesh, alpha_mesh.get_volume(), alpha_value


def pcd2binary(pcd, dpi=10):
    # dpi suggest < 20
    pcd_xyz = np.asarray(pcd.points)
    # !!!! notice !!!!
    # in numpy image system, Y axis is 0, X axis is 1
    y = pcd_xyz[:, 0]
    x = pcd_xyz[:, 1]

    x_length_m = x.max() - x.min()
    y_length_m = y.max() - y.min()
    px_num_per_cm = int(dpi / 2.54)
    width = int(np.ceil(x_length_m * 100 * px_num_per_cm))
    height = int(np.ceil(y_length_m * 100 * px_num_per_cm))
    ref_x = (x - x.min()) / x_length_m * width
    ref_y = (y - y.min()) / y_length_m * height
    ref_pos = np.vstack([ref_x, ref_y]).T.astype(int)
    ref_pos_rm_dup = np.unique(ref_pos, axis=0)

    out_img = np.zeros((width + 1, height + 1))
    out_img[ref_pos_rm_dup[:, 0], ref_pos_rm_dup[:, 1]] = 1
    out_img = out_img.astype(int)

    left_top_corner = (y.min(), x.min())

    return out_img, px_num_per_cm, left_top_corner

def get_region_props(binary, px_num_per_cm, corner):
    x_min, y_min = corner
    regions = regionprops(binary)
    props = regions[0]          # this is all coordinate in converted binary images

    # convert coordinate from binary images to real point cloud
    y0, x0 = props.centroid
    # x0, y0 = props.centroid
    center = (x0 / px_num_per_cm / 100 + x_min, y0 / px_num_per_cm / 100 + y_min)

    major_axis = props.major_axis_length / px_num_per_cm / 100
    minor_axis = props.minor_axis_length / px_num_per_cm / 100

    phi = props.orientation
    # angle = - phi * 180 / np.pi   # included angle with x axis, clockwise, by regionprops default
    # angle = phi * 180 / np.pi
    angle = np.rad2deg(phi) - 90

    return center, major_axis, minor_axis, angle, props

def get_projected_leaf_area(binary, px_num_per_cm):
    kind, number = np.unique(binary, return_counts=True)
    # back_num = number[0]
    fore_num = number[1]

    pixel_size = (1 / px_num_per_cm) ** 2   # unit is cm2

    return fore_num * pixel_size


class Broccoli(object):

    def __init__(self, pcd_input, plant_id):
        self.pcd = pcd_input
        self.plant_id = plant_id
        self.center = self.pcd.get_center()
        self.pcd_xyz = np.asarray(self.pcd.points)
        self.pcd_rgb = np.asarray(self.pcd.colors)

        # calculate the height
        self.height = self.pcd_xyz[:,2].max() - self.pcd_xyz[:,2].min()

        # calculate the convex hull 2d
        self.plane_hull, self.hull_area, _ = get_convex_hull(self.pcd, dim='2d')  # vertex_set (2D ndarray), m^2

        # calculate the convex hull 3d
        self.convex_hull3d_xy, self.hull3d_volume, self.convex_hull3d = get_convex_hull(self.pcd, dim='3d')

        # calculate the concave hull 3d
        self.concave_hull3d, self.concave_hull3d_volume, self.alpha = get_concave_hull_3d(self.pcd, alpha_st=0.005, alpha_step=0.005)

        # calculate min_area_bounding_rectangle,
        # rect_res = (rot_angle, area, width, length, center_point, corner_points)
        self.rect_res = min_bounding_rect(self.plane_hull)
        self.width = self.rect_res[2]   # unit is m
        self.length = self.rect_res[3]   # unit is m

        # calculate the projected 2D image (X-Y)
        binary, px_num_per_cm, corner = pcd2binary(self.pcd)
        # calculate region props
        self.centroid, self.major_axis, self.minor_axis, self.orient_degree, self.props = \
            get_region_props(binary, px_num_per_cm, corner)
        
        self.roundness = 4 * np.pi * self.props.area / ( self.props.perimeter**2 )

        # calculate projected leaf area
        self.pla_img = binary
        self.pla = get_projected_leaf_area(binary, px_num_per_cm) #unit is cm^2

def draw_3d_results(plant, title, save_path=None, down_voxel_size=0.005, dpi=300, show=True):
    # =-=-=-=-=-=-=-=-=-=-=-=
    # |  traits calculation |
    # -=-=-=-=-=-=-=-=-=-=-=-
    x = plant.pcd_xyz[:, 0]
    y = plant.pcd_xyz[:, 1]
    z = plant.pcd_xyz[:, 2]

    convex_hull = np.vstack([plant.plane_hull, plant.plane_hull[0, :]])
    convex_hull3d = plant.convex_hull3d

    corner = plant.rect_res[5]
    rect_corner = np.vstack([corner, corner[0, :]])

    x0 = plant.centroid[0]
    y0 = plant.centroid[1]
    phi = plant.orient_degree

    maj_x = np.cos(np.deg2rad(phi)) * 0.5 * plant.major_axis
    maj_y = np.sin(np.deg2rad(phi)) * 0.5 * plant.major_axis
    min_x = np.sin(np.deg2rad(phi)) * 0.5 * plant.minor_axis
    min_y = np.cos(np.deg2rad(phi)) * 0.5 * plant.minor_axis

    plant_vs = down_voxel_size

    downpcd = plant.pcd.voxel_down_sample(voxel_size=plant_vs)
    downpcd_xyz = np.asarray(downpcd.points)
    down_color = np.asarray(downpcd.colors)
    down_x = downpcd_xyz[:, 0]
    down_y = downpcd_xyz[:, 1]
    down_z = downpcd_xyz[:, 2]

    # ground_pcd = plant.ground_pcd.voxel_down_sample(voxel_size=plant_vs * 5)
    # ground_pcd_xyz = np.asarray(ground_pcd.points)
    # ground_x = ground_pcd_xyz[:, 0]
    # ground_y = ground_pcd_xyz[:, 1]
    # ground_z = ground_pcd_xyz[:, 2]
    # mask = ground_z < np.percentile(z, 5)
    # ground_x = ground_x[mask]
    # ground_y = ground_y[mask]
    # ground_z = ground_z[mask]

    # x_axis_max = round(max(x.max(), rect_corner[:, 0].max(), ground_x.max()), 2)
    # x_axis_min = round(min(x.min(), rect_corner[:, 0].min(), ground_x.min()), 2)
    # y_axis_max = round(max(y.max(), rect_corner[:, 1].max(), ground_y.max()), 2)
    # y_axis_min = round(min(y.min(), rect_corner[:, 1].min(), ground_y.min()), 2)
    # z_axis_max = round(max(z.max(), ground_z.max()), 2)
    # z_axis_min = round(min(z.min(), ground_z.min()), 2)
    x_axis_max = round(max(x.max(), rect_corner[:, 0].max()), 2)
    x_axis_min = round(min(x.min(), rect_corner[:, 0].min()), 2)
    y_axis_max = round(max(y.max(), rect_corner[:, 1].max()), 2)
    y_axis_min = round(min(y.min(), rect_corner[:, 1].min()), 2)
    z_axis_max = round(z.max(), 2)
    z_axis_min = round(z.min(), 2)

    # -=-=-=-=-=-=-=
    # | draw plots |
    # -=-=-=-=-=-=-=
    fig = plt.figure(figsize=(9, 6), dpi=dpi)
    ax = fig.add_subplot(111, elev=25, azim=135, box_aspect=(4,4,3), projection='3d') #box_aspect long x-axis: 8,4,3 | long y-axis: 4,10,3

    # zorder 2,3
    # # plot the ground on Y-Z
    # ax.scatter(ground_y, ground_z, zs=x_axis_max, zdir='x',
    #            color='C4', linewidth=0, marker='.', alpha=0.3, s=6, label='ground points')
    # # plot the ground on X-Z
    # ax.scatter(ground_x, ground_z, zs=y_axis_min, zdir='y',
    #            color='C4', linewidth=0, marker='.', alpha=0.3, s=6)

    # zorder 4,5,6
    # plot the shade on X-Y
    ax.scatter(down_x, down_y, zs=z_axis_min, zdir='z',
                color='C7', linewidth=0, marker='.', alpha=0.3, s=8)
    # plot the shade on Y-Z
    ax.scatter(down_y, down_z, zs=x_axis_max, zdir='x',
                color='C7', linewidth=0, marker='.', alpha=0.3, s=8)
    # plot the shade on X-Z
    ax.scatter(down_x, down_z, zs=y_axis_min, zdir='y',
                color='C7', linewidth=0, marker='.', alpha=0.3, s=8)

    # zorder 7
    # plot convex hull convert
    ax.scatter(convex_hull[:, 0], convex_hull[:, 1], color='C3', linewidth=0,
                zdir='z', zs=z_axis_min, alpha=1,
                label='convex vertex')

    # zorder 7
    # plot convex hull 3D
    for i, s in enumerate(convex_hull3d.simplices):
        s = np.append(s, s[0])
        if i == 0:
            hull_label = ax.plot(x[s], y[s], z[s], 'r-', linewidth=0.1, label='3D convex hull')
        else:
            ax.plot(x[s], y[s], z[s], 'r-', linewidth=0.1)

    # zorder 8
    # plot 3d point clouds
    ax.scatter(down_x, down_y, down_z, marker='.', color=down_color, alpha=0.7, s=8, linewidth=0, zorder=25)

    # plot convex hull
    ax.plot(convex_hull[:, 0], convex_hull[:, 1], 'C0--', alpha=0.5,
            zdir='z', zs=z_axis_min,
            label='convex hull')
    ax.plot(rect_corner[:, 0], rect_corner[:, 1], color='C1',
            alpha=0.5, zdir='z', zs=z_axis_min,
            label='min area rectangle')

    # plot the ellipse of region props
    ell = Ellipse((x0, y0), plant.major_axis, plant.minor_axis, phi,
                    alpha=0.2, zorder=0, label='region props')
    ax.add_patch(ell)
    art3d.pathpatch_2d_to_3d(ell, z=z_axis_min, zdir="z")
    # plot axis of region props
    ax.plot((x0 - maj_x, x0 + maj_x), (y0 - maj_y, y0 + maj_y), '-k',
            linewidth=2.5, zdir='z', zs=z_axis_min, zorder=7.5,
            label='ellipse axis')
    ax.plot((x0 + min_x, x0 - min_x), (y0 - min_y, y0 + min_y), '-k',
            linewidth=2.5, zdir='z', zs=z_axis_min, zorder=7.5)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    ax.set_xlim(x_axis_min, x_axis_max)
    ax.set_ylim(y_axis_min, y_axis_max)
    ax.set_zlim(z_axis_min, z_axis_max)

    ax.set_title(title, size=20)

    ax.set_box_aspect((np.ptp(x), np.ptp(y), np.ptp(z)))

    plt.legend(ncol=3, loc='lower center', bbox_to_anchor=(0.5, -0.07))
    if save_path is not None:
        plt.savefig(save_path)
    if show:
        plt.show()
        
    plt.clf()
    plt.cla()
    plt.close(fig)
    del fig, ax

def rotate_mesh(mesh_obj, R, center_xy, buffer=0.005):
    mesh_r = copy(mesh_obj).rotate(R)
    pts = np.asarray(mesh_r.vertices)

    cx, cy = center_xy

    # find all points in buffer area
    range_idx = np.where(
        (pts[:,0] > cx - buffer) & (pts[:,0] < cx + buffer) & \
        (pts[:,1] > cy - buffer) & (pts[:,1] < cy + buffer))[0]
    
    # find the highest points as z value
    max_z_idx = np.argmax(pts[range_idx, 2])
    top_points = pts[range_idx[max_z_idx], :]
    
    # get the offset_xyz
    offset_pts = np.array([cx, cy, top_points[2]])

    # move the mesh to origin
    mesh_r.vertices = o3d.utility.Vector3dVector(
        np.asarray(mesh_r.vertices) - offset_pts
    )

    return mesh_r, offset_pts


if __name__ == '__main__':
    from datetime import datetime
    import easyidp as idp
    import argparse
    import matplotlib

    matplotlib.use("Agg")  # try to solve the Fail to allocate bitmap

    mesh_database_path = r"../99_broccoli_models"

    version = f'v{datetime.now().strftime("%y%m%d")}'

    out_root = f'out/{version}/'

    findtop_folder = os.path.join(out_root, 'findtop')
    rotseg_folder = os.path.join(out_root, 'rotseg')
    easydcp_folder = os.path.join(out_root, 'easydcp')
    ztopmesh_folder = os.path.join(out_root, 'ztopmash')

    crown_index_json = os.path.join(out_root, 'crown_index.json')
    rotation_matrix_json = os.path.join(out_root, 'rotation_matrix.json')
    offset_xyz_json = os.path.join(out_root, 'offset_xyz.json')
    traits_dict_json = os.path.join(out_root, 'traits_dict.json')
    traits_csv = os.path.join(out_root, 'broccoli_traits.csv')
    
    if not os.path.exists(out_root):
        os.mkdir(out_root)
        os.mkdir(findtop_folder)
        os.mkdir(rotseg_folder)
        os.mkdir(easydcp_folder)
        os.mkdir(ztopmesh_folder)

    parser = argparse.ArgumentParser(description="Just an example",
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-a", "--all", action="store_true", help="Set mode")
    parser.add_argument("-c", "--continue", action="store_true", help="Set mode")
    parser.add_argument('-t', "--test", type=str) 
    args = parser.parse_args()
    config = vars(args)
    print(config)

    mesh_db = {}
    for file in os.listdir(mesh_database_path):
        if ".obj" in file:
            mesh_db[file[:-4]] = os.path.join(mesh_database_path, file)

    if config['continue']:
        if os.path.exists(crown_index_json):
            crown_idx = idp.jsonfile.read_json(crown_index_json)
        if os.path.exists(rotation_matrix_json):
            rotation_matrix = idp.jsonfile.read_json(rotation_matrix_json)
        if os.path.exists(offset_xyz_json):
            offset_xyz = idp.jsonfile.read_json(offset_xyz_json)
        if os.path.exists(traits_dict_json):
            traits_dict = idp.jsonfile.read_json(traits_dict_json)
    else:
        crown_idx = {}
        rotation_matrix = {}
        offset_xyz = {}
        traits_dict = {
            'id': [],
            # crown parameter
            'crown_height(m)': [],
            'center.x(m)': [], 'center.y(m)': [],
            'min_rect_width(m)': [], 'min_rect_length(m)': [], 
            'crown_2dconvex_area(cm2)': [], 'projected_leaf_area(cm2)': [],
            'centroid.x(m)': [], 'centroid.y(m)': [], 
            'long_axis(m)': [], 'short_axis(m)': [], 'orientation(degree)': [], 
            'crown_3dconvex_volume(cm3)':[], 
            'crown_3dconcave_volume(cm3)':[],
            'crown_3dconcave_alpha':[],
            'crown_roundness': [],
            # head parameter
            'head_height(m)': [],
            'head_3dconcave_volume(cm3)':[],
            'head_3dconcave_alpha':[]
        }

    pbar = tqdm(mesh_db.items(), f"Processing Meshs")
    for bro_id, bro_mesh in pbar:
        pbar.set_description(f"Processing [{bro_id}] Mesh ")
        if bro_id != config['test'] and not config['all']:
            continue

        if bro_id in crown_idx.keys() and config['continue']:
            continue

        mesh = o3d.io.read_triangle_mesh(bro_mesh, enable_post_processing=True, print_progress=False)
        def_test_pcd = mesh2pcd(mesh)

        crown_pcd, crown_colored_pcd, crown_points_index = segment_crown(def_test_pcd, 10, 0.005, show=False)

        crown_idx[bro_id] = crown_points_index

        rotated_pcd, R = rotate_to_z(
            crown_pcd, def_test_pcd, show=False, 
            save_path=f"{findtop_folder}/{bro_id}.png")
        rotation_matrix[bro_id] = R

        # draw figures
        xyz_o = np.asarray(def_test_pcd.points)
        rgb_o = np.asarray(def_test_pcd.colors)
        xyz_r = np.asarray(rotated_pcd.points)
        rgb_r = np.asarray(crown_colored_pcd.colors)
        draw_rotated(xyz_o, rgb_o, xyz_r, rgb_r, show=False, save_path=f"{rotseg_folder}/{bro_id}.png")

        crown_traits = Broccoli(rotated_pcd.select_by_index(crown_points_index), bro_id)
        head_traits = Broccoli(rotated_pcd, bro_id)

        draw_3d_results(crown_traits, bro_id, down_voxel_size=0.001, save_path=f"{easydcp_folder}/{bro_id}.png", show=False)

        traits_dict['id'].append(bro_id)
        # crown_traits
        traits_dict['crown_height(m)'].append(crown_traits.height)
        traits_dict['center.x(m)'].append(crown_traits.center[0])
        traits_dict['center.y(m)'].append(crown_traits.center[1])
        traits_dict['min_rect_width(m)'].append(crown_traits.width)
        traits_dict['min_rect_length(m)'].append(crown_traits.length)
        traits_dict['crown_2dconvex_area(cm2)'].append(crown_traits.hull_area * 100 * 100)
        traits_dict['projected_leaf_area(cm2)'].append(crown_traits.pla)
        traits_dict['centroid.x(m)'].append(crown_traits.centroid[0])
        traits_dict['centroid.y(m)'].append(crown_traits.centroid[1])
        traits_dict['long_axis(m)'].append(crown_traits.major_axis)
        traits_dict['short_axis(m)'].append(crown_traits.minor_axis)
        traits_dict['orientation(degree)'].append(crown_traits.orient_degree)
        traits_dict['crown_roundness'].append(crown_traits.roundness)

        traits_dict['crown_3dconvex_volume(cm3)'].append(crown_traits.hull3d_volume * 1000 * 1000)
        traits_dict['crown_3dconcave_volume(cm3)'].append(crown_traits.concave_hull3d_volume * 1000 * 1000)
        traits_dict['crown_3dconcave_alpha'].append(crown_traits.alpha)

        # head_traits
        traits_dict['head_height(m)'].append(head_traits.height)
        traits_dict['head_3dconcave_volume(cm3)'].append(head_traits.concave_hull3d_volume * 1000 * 1000)
        traits_dict['head_3dconcave_alpha'].append(head_traits.alpha)

        # save the rotated mesh
        out_mesh, offset_pts = rotate_mesh(mesh, R, crown_traits.centroid, 0.002)
        o3d.io.write_triangle_mesh(f'{ztopmesh_folder}/{bro_id}.obj', out_mesh, print_progress=False)

        offset_xyz[bro_id] = offset_pts

        idp.jsonfile.dict2json(crown_idx, crown_index_json)
        idp.jsonfile.dict2json(rotation_matrix, rotation_matrix_json)
        idp.jsonfile.dict2json(offset_xyz, offset_xyz_json)
        idp.jsonfile.dict2json(traits_dict, traits_dict_json)

        if not config['all']:
            break

    out_pd = pd.DataFrame(traits_dict)
    out_pd.to_csv(traits_csv, index=False)