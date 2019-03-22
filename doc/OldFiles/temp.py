while(capture.isOpened()):
    Image_Batch_Size=16
    road_frame_S=[]
    error_Found=False
    while(len(road_frame_S)<Image_Batch_Size):
        ret, temp_frame = capture.read()
        if ret:
            road_frame_S.append(temp_frame)
        else:
            error_Found=True;
            break
    if error_Found:
        break
    results = model.detect(road_frame_S, verbose=0)
    road_frame_s_output=[]
    for index,r in enumerate(results):
        road_frame_output= display_instances(
                road_frame_S[index], r['rois'], r['masks'], r['class_ids'], class_names, r['scores']
            )
        output.write(road_frame_output)
        out_dict[count] = {}
        out_dict[count]['cls_id'] = r['class_ids']
        out_dict[count]['rois'] = r['rois']
        print('[INFO] Frame {}/{}'.format(count,frameCount))
        count += 1
