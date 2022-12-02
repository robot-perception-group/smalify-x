import csv

def frame_to_unix(frame, sec_per_frame, vid_trim_sync, unix_sync, vid_pose_sync):
    return frame*sec_per_frame + vid_trim_sync + unix_sync - vid_pose_sync

def frame_to_pose(frame, csv_fn, sec_per_frame, vid_trim_sync, unix_sync, vid_pose_sync):
    frame_unix_time = round(frame_to_unix(frame, sec_per_frame, vid_trim_sync, unix_sync, vid_pose_sync),1)
    with open(csv_fn) as csv_file:
            csv_reader = csv.DictReader(csv_file)
            for row in csv_reader:
                row_unix_time = float(row['Time(sec)']) + float(row[' Time(n_sec)'])*1e-9
                #print(row_unix_time, frame_unix_time)
                if row_unix_time == frame_unix_time:
                    return row