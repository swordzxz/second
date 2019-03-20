# time process

## all time: 130ms


def velo_callback(msg):
## read_points time : 56ms
    arr_bbox = BoundingBoxArray()
#####1.read_points time : 0.000001ms
    pcl_msg = pc2.read_points(msg, skip_nans=False, field_names=("x", "y", "z","intensity")) 
#####2. points2array time : 50ms    
    np_p = np.array(list(pcl_msg), dtype=np.float32)
#####3. create_cond time : 4ms
    cond = hv_in_range(x=np_p[:, 0],
                       y=np_p[:, 1],
                       z=np_p[:, 2],
                       fov=[-35, 35],
                       fov_type='h')
#####4.cut_poinst time : 0.4ms  24000*4
    np_p_ranged = np_p[cond]

## net_predict time : 49ms
    results = test(np_p_ranged,
                   net=net,
                   voxel_generator=voxel_generator,
                   target_assigner=target_assigner,
                   result_path_step=result_path_step,
                   model_cfg=model_cfg,
                   center_limit_range=center_limit_range
         )

## publish time : 21ms
#####1. create_arrbox time : 0.00008ms
    if len(results[0]) != 0:

        for result in results[0]:
            bbox = BoundingBox()

            bbox.header.frame_id = msg.header.frame_id
            q = quaternion_from_euler(0,0,-float(result[7]))

            bbox.pose.orientation.x = q[0];
            bbox.pose.orientation.y = q[1];
            bbox.pose.orientation.z = q[2];
            bbox.pose.orientation.w = q[3];
            bbox.pose.position.x = float(result[1])
            bbox.pose.position.y = float(result[2])
            bbox.pose.position.z = float(result[3])
            bbox.dimensions.x = float(result[6])
            bbox.dimensions.y = float(result[5])
            bbox.dimensions.z = float(result[4])
            
            arr_bbox.boxes.append(bbox)

    arr_bbox.header.frame_id = msg.header.frame_id


    if len(arr_bbox.boxes) is not 0: 
#####2. pubulish_modified_points time : 21ms
        publish_test(np_p_ranged, msg.header.frame_id)
#####3. publish_arrbox time : 0.000006ms
        pub_arr_bbox.publish(arr_bbox)
        arr_bbox.boxes.clear()
        

def publish_test(np_p_ranged,frame_id):
    header = Header()
    header.stamp = rospy.Time()
    header.frame_id = frame_id
    x = np_p_ranged[:, 0].reshape(-1)
    y = np_p_ranged[:, 1].reshape(-1)
    z = np_p_ranged[:, 2].reshape(-1)
    if np_p_ranged.shape[1] == 4:
        i = np_p_ranged[:, 3].reshape(-1)
    else:
        i = np.zeros((np_p_ranged.shape[0], 1)).reshape(-1)
    cloud = np.stack((x, y, z, i)
    msg_segment = pc2.create_cloud(header=header,
                                    fields=_make_point_field(4),
                                    points=cloud.T)
    pub_velo.publish(msg_segment) 