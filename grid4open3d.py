import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
np.set_printoptions(suppress=True)
import pandas as pd
import seaborn as sns
import open3d as o3d

from shapely.geometry.polygon import Polygon

import easyidp as idp

from broccoli_file_path import *

def to_open3d(idp_pcd):
    final_pcd = o3d.geometry.PointCloud()
    final_pcd.points = o3d.utility.Vector3dVector(idp_pcd._points)
    color_np = idp_pcd.colors/255
    # color_np = np.insert(color_np, 2, 0.5, axis=1)
    # print(color_np)
    final_pcd.colors = o3d.utility.Vector3dVector(color_np)

    return final_pcd

if __name__ == '__main__':
    date_loop = True
    while date_loop:
        date = input(f'select a date to preview {list(process_date.keys())}:\n>>> ')

        if date not in process_date.keys():
            print(f"[io][{date}] not in available date")
        else:
            date_loop = False
    print('\nLoading data...')
    grid = idp.ROI(grid_shp, name_field='id')

    print('loading field point cloud')
    pcd = idp.PointCloud(process_date[date]['pcd'])

    print('loading UAV 2D trait results')
    broccoli_2d_traits = pd.read_excel('./out/props_all_2022.xlsx')

    print('loading AutoML prediction models')
    # load autosklearn model
    with open('out/autoskl_model_1.pkl', 'rb') as handle:
        automl = pickle.load(handle)

    # load the database
    print('loading HD 3D broccoli head database')
    broccoli_mesh_db = pd.read_excel('out/autoskl_model_db.xlsx')

    mesh_database_path = r"out/v230407/ztopmash"
    mesh_db = {}
    for file in os.listdir(mesh_database_path):
        if ".obj" in file:
            mesh_db[file[:-4]] = os.path.join(mesh_database_path, file)

    # select grids
    while True:
        grid_id = input('specify the grid id want to show:\n>>> ')

        if grid_id not in grid.keys():
            print('wrong grid id, please try again')
            continue
            
        grid_pcd = pcd.crop_point_cloud(grid[grid_id])
        grid_ctr = (grid_pcd._points.max(axis=0) + grid_pcd._points.min(axis=0)) / 2

        ################################
        # read brococli head positions #
        ################################

        ## find inside the grid bbox
        grid_xymin = grid[grid_id].min(axis=0)
        grid_xymax = grid[grid_id].max(axis=0)

        in_grid_idx = (broccoli_2d_traits.date == f"20{date}_0") & \
        (broccoli_2d_traits.circular_x > grid_xymin[0]) & (broccoli_2d_traits.circular_x < grid_xymax[0]) & \
        (broccoli_2d_traits.circular_y > grid_xymin[1]) & (broccoli_2d_traits.circular_y < grid_xymax[1])

        broccoli_2d_in_grid = broccoli_2d_traits[in_grid_idx]
        ## find insde the grid polygon
        grid_polygon = Polygon(grid[grid_id])
        in_grid_poly_inx = []
        for i, val in broccoli_2d_in_grid.iterrows():
            bro_head_polygon = Polygon(eval(val.polygon))

            in_grid_poly_inx.append( grid_polygon.contains(bro_head_polygon) )

        broccoli_2d_in_grid_poly = broccoli_2d_in_grid[in_grid_poly_inx].reset_index()

        # make the matrix for auto-sklearn
        pd.set_option('mode.chained_assignment',None) # supress SettingWithCopyWarning

        X = broccoli_2d_in_grid_poly[[
            'major_axis_length', 'minor_axis_length', 
            'min_area_rect_min', 'min_area_rect_max',
            'area', 'convex_area',
        ]]
        X.major_axis_length /= 1000
        X.minor_axis_length /= 1000
        X.min_area_rect_max /= 1000
        X.min_area_rect_min /= 1000
        X.area /= 100
        X.convex_area /= 100

        # predict the values
        predictions = automl.predict(X.values)
        predictions = pd.DataFrame(columns=X.columns, data=predictions)

        ## preview the distribution of predicted tratis
        sns.set_theme(style="ticks")

        predictions4plot = predictions.copy()
        predictions4plot.area /= 1000
        predictions4plot.convex_area /= 1000

        predictions4plot = predictions4plot.rename(columns={
            'major_axis_length': 'length of\nmajor axis\n(m)', 
            'minor_axis_length': 'length of\nminor axis\n(m)', 
            'min_area_rect_min': 'width of\nmin. area\nrect.(m)',
            'min_area_rect_max': 'length of\nmin. area\nrect.(m)', 
            'area': 'area\n($1000 cm^2$)', 
            'convex_area': 'convex area\n($1000 cm^2$)',
        })

        f, ax = plt.subplots(figsize=(7, 6))

        sns.boxplot(data=predictions4plot, ax=ax, palette="vlag")
        sns.stripplot(data=predictions4plot, ax=ax)

        # plt.xticks(rotation=-45)

        ax.set_title(f'AutoML predicted traits of grid {grid_id} on {date}')
        ax.set_ylabel('values')

        plt.tight_layout()
        plt.savefig(f'out/autoskl_pred_{date}_{grid_id}.svg')
        plt.show()

        #######################
        # find the closest db #
        #######################

        print('finding the closest templates, this may cost a few minutes...')

        # prepare matrix for matching
        matching_sort = broccoli_mesh_db[[
            'long_axis(m)', 'short_axis(m)', # 'crown_roundness',
            'min_rect_width(m)', 'min_rect_length(m)', 
            'crown_2dconvex_area(cm2)', 'projected_leaf_area(cm2)']]

        # preview the match results
        fig, ax = plt.subplots(1,1, figsize=(8,8))

        trans = []
        matched_idx = []

        lbmax = np.ceil( 0.01 * matching_sort.values.max()) / 10

        ax.plot([0, lbmax], [0, lbmax], 'k--', alpha=0.5)

        for row_id, v in predictions.iterrows():
            diff = matching_sort.values - v.values
            # normalize
            diff_norm = np.abs(diff / np.linalg.norm(diff, axis=0))
            diff_sum = diff_norm.sum(axis=1)

            min_diff_idx = np.argmin(diff_sum)

            pred_v = v.values
            db_v = matching_sort.values[min_diff_idx,:]

            plot_ratio = np.array([1, 1, 1, 1, 0.001, 0.001])
            ax.scatter(x=pred_v*plot_ratio, y=db_v*plot_ratio, alpha=0.5)

            # [2]'min_rect_width(m)', [3] 'min_rect_length(m)', 
            trans.append((pred_v[2]/db_v[2], pred_v[3]/db_v[3]))
            matched_idx.append(min_diff_idx)
            
        ax.set_xlim(0, lbmax)
        ax.set_ylim(0, lbmax)

        ax.text(lbmax*0.01, lbmax*0.99, 
            'Traits Units \n\n'
            'length of major axis (m)\n' + 
            'length of minor axis (m)\n' + 
            'width of min. area rect. (m)\n' + 
            'length of min. area rect. (m)\n'+ 
            'area ($1000cm^2$)\n' + 
            'convex area ($1000cm^2$)\n',
            va='top')
        ax.set_xlabel('Traits of AutoML predicted')
        ax.set_ylabel('Traits of matched broccoli template')
        ax.set_title(f'Date {date} Grid ID {grid_id}')
        plt.tight_layout()
        plt.savefig(f"out/autoskl_matched_{date}_{grid_id}.svg")
        plt.show()

        # put broccoli back to field
        broccoli_mesh_container = []
        for i, row in broccoli_2d_in_grid_poly.iterrows():
            polygon_np = np.array(eval(row.polygon))
            polygon_xyz = grid_pcd.crop_point_cloud(polygon_np)
            ht_values = polygon_xyz.points[:,2]

            # 95% percentile ht
            thresh = np.percentile(ht_values, 95, axis=0)
            z_pos = np.mean(ht_values[ht_values >= thresh])

            # offseted
            x_pos = row.circular_x - grid_pcd.offset[0]
            y_pos = row.circular_y - grid_pcd.offset[1]
            z_pos -= grid_pcd.offset[2]

            # find
            matched_slice = broccoli_mesh_db.iloc[matched_idx[i]]
            mesh = o3d.io.read_triangle_mesh(mesh_db[matched_slice.Model_ID],  enable_post_processing=True)

            # rotate and put it back
            r = Rotation.from_euler('z', matched_slice['orientation(degree)'], degrees=True)
            m_trans = copy(mesh).rotate(r.as_matrix())

            # scale
            m_trans.vertices = o3d.utility.Vector3dVector(
                np.asarray(m_trans.vertices) * np.array([trans[i][0], trans[i][1], 1]) 
            )
            # by testing, the x -> width, y -> length
            # trans -> # [0]'min_rect_width(m)', [1] 'min_rect_length(m)', 
            m_trans_back = copy(m_trans).rotate(r.inv().as_matrix())

            broccoli_mesh_container.append(m_trans_back.translate((x_pos, y_pos, z_pos)))

        # open3d show
        grid_pcd_o3d = to_open3d(grid_pcd)
        o3d.visualization.draw_geometries([grid_pcd_o3d] + broccoli_mesh_container)