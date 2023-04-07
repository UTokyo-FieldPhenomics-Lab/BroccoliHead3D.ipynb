from utils230407 import *

# the version only recalcualte easydcp traits and rotate meshes by min_bbox.center


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
            save_path=None)
        rotation_matrix[bro_id] = R

        # draw figures
        # xyz_o = np.asarray(def_test_pcd.points)
        # rgb_o = np.asarray(def_test_pcd.colors)
        # xyz_r = np.asarray(rotated_pcd.points)
        # rgb_r = np.asarray(crown_colored_pcd.colors)
        # draw_rotated(xyz_o, rgb_o, xyz_r, rgb_r, show=False, save_path=f"{rotseg_folder}/{bro_id}.png")

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
        # out_mesh, offset_pts = rotate_mesh(mesh, R, crown_traits.rect_res[4], 0.002)
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