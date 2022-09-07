import numpy as np
import cv2 as cv
import json
import matplotlib.pyplot as plt

# target camera
cams = [0,1,2,3,4,5,6,7]
N_views_total = len(cams)

# target frames
frames = [110]

# skip bad objects
skip_id = [301, 303]
colors = [0, 'red', 'green', 'blue', 'yellow', 'cyan', 'magenta', '#ae7181', 'black', '#13eac9', '#cb416b', '#6b8ba4', '#61de2a', '#0c06f7', '#0a5f38', '#d99b82']

def readCameraP0_as_np_tanks(cameraPO_file):

    with open(cameraPO_file) as f:
        lines = f.readlines() 
    cameraRTO = np.empty((3, 4)).astype(np.float64)
    cameraRTO[0,:] = np.array(lines[1].rstrip().split(' ')[:4], dtype=np.float64)
    cameraRTO[1,:] = np.array(lines[2].rstrip().split(' ')[:4], dtype=np.float64)
    cameraRTO[2,:] = np.array(lines[3].rstrip().split(' ')[:4], dtype=np.float64)
    
    cameraKO = np.empty((3, 3)).astype(np.float64)
    cameraKO[0,:] = np.array(lines[7].rstrip().split(' ')[:3], dtype=np.float64)
    cameraKO[1,:] = np.array(lines[8].rstrip().split(' ')[:3], dtype=np.float64)
    cameraKO[2,:] = np.array(lines[9].rstrip().split(' ')[:3], dtype=np.float64)
    

    cameraPO = np.dot(cameraKO, cameraRTO)
    return cameraPO, cameraRTO, cameraKO

# reference view bbox四个点与 N_view-1 个 source view 的射线相交
def calculate_3d_intersect(x1, K1, Ks, R1, Rs, T1, Ts, ls):
    r1t1 = np.linalg.inv(R1).dot(T1.T)

    s1 = np.matmul(Ks, np.matmul(Rs, np.linalg.inv(R1).dot(T1.T)[None, :, :]))  # (N_view-1,3,1)
    s2 = np.matmul(Ks, Ts.transpose(0, 2, 1))  # (N_view-1,3,1)
    rkx1 = np.matmul(np.linalg.inv(R1).dot(np.linalg.inv(K1))[None, ...], x1[:, :, None])  # (4,3,1)
    s3 = np.matmul(Ks[None, ...], np.matmul(Rs[None, ...], rkx1[:, None, :, :]))  # (4,N_view-1,3,1)

    up = np.matmul(ls[:, None, :, None, :], (s1 - s2)[None, None, ...])[..., 0]  # (2,1,N_view-1,1)
    down = np.matmul(ls[:, None, :, None, :], s3[None, ...])[..., 0]  # (2,4,N_view-1,1)
    lamda1 = up / (down + 1e-10)  # (2,4,N_view,1)

    k_lamda_x = np.matmul(np.linalg.inv(K1)[None, None, None, ...],
                          (lamda1 * x1[None, :, None, :])[..., None])  # (2,4,N_view-1,3,1)
    X1 = np.matmul(np.linalg.inv(R1), k_lamda_x - T1[None, None, ..., None])  # (2,4,N_view-1,3,1)
    X1 = X1.reshape(-1, 3)

    return X1

# ### 求交函数

def projection_reject(X1, pts_image_predict, tr,br,tl,bl, N_view, ratio = 0.5):
    '''
        X1: (N_points, 3)
        pts_image_predict:(N_view, N_points, 2)
        ratio: adjust reject region size. Larger means larger size
    return:
        X1_new:(N_point_new, 3)
    '''
    #pt = pts_image_predict[...,[1,0]]
    pt = pts_image_predict
    #pt = pts_image_predict[...,[1,0]]
    #pts_image_predict[...,1] = img[0].shape[0] - pts_image_predict[...,1]
    fix_pixel = 50

    gap_0 = fix_pixel + ratio * (tr[:,0][:,None] - tl[:,0][:,None])
    gap_1 = fix_pixel + ratio * (tr[:,1][:,None] - br[:,1][:,None])
    #gap_0 = gap_1 = ratio
    #pdb.set_trace()
    #print('gap', gap_0, gap_1)
    pt_mask = (pt[...,0] <= (tr[:,0][:,None]+gap_0))*(pt[...,0] >= (tl[:,0][:,None]-gap_0))*(pt[...,1] <= (tr[:,1][:,None]+gap_1))*(pt[...,1] >= (br[:,1][:,None]-gap_1)) #(N_view, N_points)
    # print(pt_mask)
    pt_mask = pt_mask.sum(axis = 0) > (N_view-1)
    X1_new = X1[pt_mask,:]
    # X1_new = X1
    return X1_new

def projection_image(P_gt, pts):
    '''
    P_gt:(N_view, 3, 4)
    pts:(N,3)
    
    return: (N_view, N, 2)
    '''
    P = np.concatenate((pts, np.ones((pts.shape[0],1))), axis = 1) #(N,4)
    #pdb.set_trace()
    pts_image = np.matmul(P_gt[:,None,:,:], P[None,:,:,None])[...,0] #(N_view,N, 3)
    pts_image = pts_image[...,:2]/pts_image[...,2:]
    pts_image = np.int32(pts_image)

    return pts_image

def count_3D_point(tr, tl, br, bl, K_total, R_total, T_total, line_right, line_left, ratio_bbox = 0.5):
    '''
        tr, tl, br, bl: bbox  ; shape:(N_view, 2)
        K_total, R_total, T_total: camera position
        line_right, line_left: lines
    '''
    P_gt = np.matmul(K_total, np.concatenate((R_total, T_total.transpose(0,2,1)), axis=2))
    N_view = tr.shape[0]
    X1_new_total = None
    for i in range(1):
        x1 = np.concatenate((tr[i:(i+1)], tl[i:(i+1)], br[i:(i+1)], bl[i:(i+1)]), axis = 0)
        x1 = np.concatenate((x1, np.ones((x1.shape[0],1))), axis = 1) #(4,3)
        x1_mid = x1.mean(axis = 0)[None,:]

        x1 = np.concatenate((x1, x1_mid), axis = 0)
        
        K1, Ks = K_total[i], np.delete(K_total, i, 0) #(3,3)(N_view-1,3,3)
        R1, Rs = R_total[i], np.delete(R_total, i, 0) #(3,3)(N_view-1,3,3)
        T1, Ts = T_total[i], np.delete(T_total, i, 0) #(1,3)(N_view-1,1,3)

        inter_num = 2
        ratio = np.linspace(0.0,1.0,inter_num)
        ls = np.delete(line_right[None,...], i, 1) * ratio[:,None,None] + np.delete(line_left[None,...], i, 1) * (1 - ratio[:,None,None])
        X1 = calculate_3d_intersect(x1, K1, Ks, R1, Rs, T1, Ts, ls)
        pts_image_predict = projection_image(P_gt, X1)

        X1_new = projection_reject(X1, pts_image_predict, tr,br,tl,bl, N_view, ratio = ratio_bbox)
        X1_new_total = X1_new if X1_new_total is None else np.concatenate((X1_new_total, X1_new), axis = 0)

    return X1_new_total


def count_3D_bbox_point(tr, tl, br, bl, K_total, R_total, T_total, line_right, line_left, ratio_bbox=0.5):
    '''
        tr, tl, br, bl: bbox  ; shape:(N_view, 2)
        K_total, R_total, T_total: camera position
        line_right, line_left: lines
    '''
    P_gt = np.matmul(K_total, np.concatenate((R_total, T_total.transpose(0, 2, 1)), axis=2))
    N_view = tr.shape[0]
    X1_new_total = None
    for i in range(N_view):
        # if i != 2:
        #     continue
        x1 = np.concatenate((tr[i:(i + 1)], tl[i:(i + 1)], br[i:(i + 1)], bl[i:(i + 1)]), axis=0)
        x1 = np.concatenate((x1, np.ones((x1.shape[0], 1))), axis=1)  # (4,3)
        x1_tb = np.concatenate((x1[0:2].mean(axis=0)[None, :], x1[2:4].mean(axis=0)[None, :]), axis=0)
        x1_mid = x1.mean(axis=0)[None, :]

        x1 = np.concatenate((x1, x1_mid), axis=0)

        K1, Ks = K_total[i], np.delete(K_total, i, 0)  # (3,3)(N_view-1,3,3)
        R1, Rs = R_total[i], np.delete(R_total, i, 0)  # (3,3)(N_view-1,3,3)
        T1, Ts = T_total[i], np.delete(T_total, i, 0)  # (1,3)(N_view-1,1,3)

        inter_num = 2
        ratio = np.linspace(0.0, 1.0, inter_num)
        ls = np.delete(line_right[None, ...], i, 1) * ratio[:, None, None] + np.delete(line_left[None, ...], i, 1) * (
                    1 - ratio[:, None, None])

        X1 = calculate_3d_intersect(x1, K1, Ks, R1, Rs, T1, Ts, ls)
        pts_image_predict = projection_image(P_gt, X1)
        X1_new = projection_reject(X1, pts_image_predict, tr, br, tl, bl, N_view, ratio=ratio_bbox)
        X1_new_total = X1_new if X1_new_total is None else np.concatenate((X1_new_total, X1_new), axis=0)
    return X1_new_total

def show_projection(pts_image_predict, tr,br,tl,bl ):
    N_view = tr.shape[0]
    # plot the rig
    fig, axs = plt.subplots(1, N_view, figsize=(40, 5))

    for i in range(N_view):

        axs[i].scatter(pts_image_predict[i,:,0], pts_image_predict[i,:,1], c = 'red')
        axs[i].plot([tr[i,0],tl[i,0]],[tr[i,1],tl[i,1]], c = 'red')
        axs[i].plot([tl[i,0],bl[i,0]],[tl[i,1],bl[i,1]], c = 'red')
        axs[i].plot([bl[i,0],br[i,0]],[bl[i,1],br[i,1]], c = 'red')
        axs[i].plot([br[i,0],tr[i,0]],[br[i,1],tr[i,1]], c = 'red')
        axs[i].imshow(img[i])

    plt.show()


###############

def plot_camera_3d(ax_3d, R_inv, T_inv, K, img_w = 5000, img_h = 3000):

    plot_point = np.array([[0,0,0],[img_w, img_h, 1], [img_w, 0, 1], [0, img_h, 1], [0, 0, 1]]) #(5,3)

    plot_point = plot_point[:,None,...][...,None]#(5,1,3,1)

    for i in range(R_inv.shape[0]):
        K_inv = np.linalg.inv(K[i])[None,None]
        plot_point_glob = np.matmul(R_inv[None,[i]], np.matmul(K_inv,plot_point)) + T_inv[None, [i],:, None]  # (5,1,3,1)

        p_i_0 = plot_point_glob[0,0,:,0] #(3)
        p_i_1s = plot_point_glob[1:, 0, :, 0]  # (4,3)
        for j in range(4):
            ax_3d.plot([p_i_0[0], p_i_1s[j,0]], [p_i_0[1], p_i_1s[j,1]], [p_i_0[2], p_i_1s[j,2]], c = 'r')


def mode_num(arr):
    dict_cnt = {}
    for x in arr:
        dict_cnt[x] = dict_cnt.get(x,0) + 1
    max_cnt = max(dict_cnt.values())
    most_values = [k for k,v in dict_cnt.items() if v==max_cnt]
    return most_values[0]


if __name__ == '__main__':
    
    annotation_3d = {}
    # ## 数据导入

    # 导入数据标注
    custom_annotation = json.load(open('./annotations/annotation.json'))
    custom_data_info = json.load(open('./annotations/data_info.json'))

    # parameters
    ind_to_classes = custom_data_info['ind_to_classes']
    ind_to_predicates = custom_data_info['ind_to_predicates']

    # ### 导入相机位姿
    RT = []
    K = []


    for cam_id in range(0, N_views_total):
        id_form = "{:0>2d}".format(cam_id)
        _, RT_temp, K_temp = readCameraP0_as_np_tanks('./calibration/TPV-' + id_form + "-calib.txt")
        RT.append(RT_temp)
        K.append(K_temp)

    RT_temp = np.array(RT)
    R_inv = np.array([R.T for R in RT_temp[:,:,:3]])
    T_inv = np.array([np.matmul(-R.T, T) for R, T in zip(RT_temp[:,:,:3], RT_temp[:,:,3])])

    # main

    for frame in frames:
        # 画出相机坐标轴, fig1, ax1
        coord_points = []
        N_points = 20
        step = 0.05
        for j in range(3):
            for i in range(0,N_points):
                point = np.zeros(3)
                point[j] = i*step
                coord_points.append(point)
        coord_points = np.array(coord_points)

        fig_3d = plt.figure(figsize=(8, 6))
        ax_3d = fig_3d.add_subplot(111, projection='3d')
        ax_3d.set_xlim3d(-1,6)
        ax_3d.set_ylim3d(-4,3)
        ax_3d.set_zlim3d(-1,6)

        fig_bbox, ax_bbox = plt.subplots(1, N_views_total, figsize=(40, 5))
        for i in range(N_views_total):
            ax_bbox[i].invert_yaxis()

        current_frame = {'bbox_3d': [], 'instance_id': [], 'bbox_labels': [], 'rel_pair': [], 'rel_label': []}
        boxes = []
        instance_id = []
        bbox_labels = []
        rel_pairs = []
        rel_labels = []
        img_list = []
        rel_labels_3d = []

        for cam_id in range(N_views_total):

            idx = "{:0>2d}".format(cam_id)

            # ### 导入图像
            img_path = custom_data_info['ind_to_files'][str(frame)][idx]
            img_path = img_path.replace('\\', '/')
            img_list.append(cv.imread(img_path))

            # ### 导入 bbox & label
            boxes.append(custom_annotation[str(frame)][idx]['bbox'][:])
            instance_id.append(custom_annotation[str(frame)][idx]['instance_id'][:])
            bbox_labels.append(custom_annotation[str(frame)][idx]['bbox_label'][:])

            rel_pairs.append(custom_annotation[str(frame)][idx]['rel_pair'])
            rel_labels.append(custom_annotation[str(frame)][idx]['rel_label'])
        
        unique_id = np.unique([id for id_list in instance_id for id in id_list])
        unique_rel = []
        for pair_list in rel_pairs:
            for pair in pair_list:
                if pair not in unique_rel:
                    unique_rel.append(pair)

        for rel_pair in unique_rel:
            rel_label_list = []
            for cam_id in range(N_views_total):
                for loc, pair in enumerate(rel_pairs[cam_id]):
                    if (np.all(pair == rel_pair)):
                        rel_label_list.append(rel_labels[cam_id][loc])
            rel_labels_3d.append(mode_num(rel_label_list))
        current_frame['rel_pair'].append(unique_rel)
        current_frame['rel_label'].append(rel_labels_3d)

        for bbox_id in unique_id:
            if bbox_id in skip_id:
                continue
            # 在各个图像检索 2d bbox：

            loc_list = [np.where(np.array(instance_id[cam_idx]) == bbox_id)[0] for cam_idx in range(N_views_total)]
            loc_list = [loc[0] if np.size(loc) else None for loc in loc_list]
            if (sum(loc_list != np.array(None)) < 2):
                print("Skip " + str(bbox_id))
                continue

            # ### 导入 bbox 和对应相机
            RT_total = np.empty(0)
            K_total = np.empty(0)
            bbox_current_id = []
            bbox_label = []
            img = []
            for cam_id in range(0, N_views_total):
                if (loc_list[cam_id] != None):
                    if (RT_total.any()):
                        RT_total = np.concatenate((RT_total, RT[cam_id][None,...]), axis = 0)
                        K_total = np.concatenate((K_total, K[cam_id][None,...]), axis = 0)
                    else:
                        RT_total = RT[cam_id][None,...]
                        K_total = K[cam_id][None,...]
                
                    bbox_current_id.append(boxes[cam_id][loc_list[cam_id]])
                    bbox_label.append(bbox_labels[cam_id][loc_list[cam_id]])
                    img.append(img_list[cam_id])
            N_views_visible = len(bbox_current_id)
            R_total = RT_total[:,:,:3]
            T_total = RT_total[:,:,3][:,None,:]

            P_gt = np.matmul(K_total, np.concatenate((R_total, T_total.transpose(0,2,1)), axis=2))


            #生成四个角的坐标：
            bbox_current_id = np.array(bbox_current_id)
            bl = bbox_current_id[:, 0:2].copy()
            tr = bbox_current_id[:, 2:4].copy()
            tl = np.concatenate((bl[:, 0:1], tr[:, 1:2]), axis=1)
            br = np.concatenate((tr[:, 0:1], bl[:, 1:2]), axis=1)

            # ## 计算交点 

            # ### 2D bbox线段方程
            ones = np.ones((tr.shape[0],1))
            line_right = np.concatenate((-ones,0*ones,tr[:,0:1]), axis=1)
            line_left = np.concatenate((-ones,0*ones,tl[:,0:1]), axis=1)

            # ### 求解交点
            X1_new_total = count_3D_bbox_point(tr, tl, br, bl, K_total, R_total, T_total,
                                               line_right, line_left, ratio_bbox = 0.3)
            
            pts_image_predict = projection_image(P_gt, X1_new_total)

            # 跳过 3D 点个数少于 2 的 instance
            if (len(X1_new_total) < 2):
                print("Skip " + str(bbox_id))
                continue

            # #### 从相机坐标重新变化回图像坐标点
            tr_shift = tr.copy()
            br_shift = br.copy()
            tl_shift = tl.copy()
            bl_shift = bl.copy()

            x_min = min(X1_new_total[:,0])
            x_max = max(X1_new_total[:,0])
            y_min = min(X1_new_total[:,1])
            y_max = max(X1_new_total[:,1])
            z_min = min(X1_new_total[:,2])
            z_max = max(X1_new_total[:,2])

            h = z_max-z_min
            w = y_max-y_min
            l = x_max-x_min
            tx = (x_max+x_min)/2
            ty = (y_max+y_min)/2
            tz = (z_max+z_min)/2

            bbox_3d = [h,w,l,tx,ty,tz]
            bbox_point = np.array([[x, y, z] for z in [z_min, z_max] for y in [y_min, y_max] for x in [x_min, x_max]])

            line_pairs = [[0,1], [2,3], [4,5], [6,7], [0,2], [1,3], [4,6], [5,7], [0,4], [1,5], [2,6], [3,7]]
            color = colors[bbox_label[0]]

            # 画出 3D bbox
            for line in line_pairs:
                ax_3d.plot([bbox_point[line[0],0],bbox_point[line[1],0]],[bbox_point[line[0],1],bbox_point[line[1],1]], [bbox_point[line[0],2],bbox_point[line[1],2]], c = color)

            bbox_corner = projection_image(P_gt, bbox_point)

            for i in range(N_views_visible):
                for line in line_pairs:
                    ax_bbox[i].plot([bbox_corner[i,line[0],0],bbox_corner[i,line[1],0]],[bbox_corner[i,line[0],1],bbox_corner[i,line[1],1]], c = color)

            current_frame['bbox_3d'].append(bbox_3d)
            current_frame['instance_id'].append(bbox_id)
            current_frame['bbox_labels'].append(bbox_label[0])

        annotation_3d[str(frame)] = current_frame
        for i, pair in enumerate(unique_rel):
            color = colors[rel_labels_3d[i]]
            loc1 = np.where(np.array(current_frame['instance_id'])==pair[0])[0]
            loc2 = np.where(np.array(current_frame['instance_id'])==pair[1])[0]
            if np.size(loc1) and np.size(loc2):
                ax_3d.plot([current_frame['bbox_3d'][loc1[0]][3], current_frame['bbox_3d'][loc2[0]][3]],[current_frame['bbox_3d'][loc1[0]][4], current_frame['bbox_3d'][loc2[0]][4]],[current_frame['bbox_3d'][loc1[0]][5], current_frame['bbox_3d'][loc2[0]][5]], c=color)
        
        for cam_id in range(N_views_total):
            ax_bbox[cam_id].imshow(img_list[cam_id])

        plt.show()
        
    class MyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            else:
                return super(MyEncoder, self).default(obj)

    out_path = "./annotations/annotation_3D.json"
    with open(out_path,'w') as f:
        json.dump(annotation_3d, f, ensure_ascii=False, cls=MyEncoder)
