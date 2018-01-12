Executable files:
darknet/label_gpu
darknet/tracking
darknet_label_skx/label_skx


To run label:

idev -N 1 -n 1 -p vis

cd /work/00410/huang/maverick/yolo/latest_yolo/darknet_dbg_0/job_benchmark/video_3
ln -s /work/00410/huang/maverick/yolo/latest_yolo/darknet_dbg_0/cfg .
ln -s /work/00410/huang/maverick/yolo/latest_yolo/darknet_dbg_0/data .
ln -s /work/00410/huang/maverick/yolo/yolo.weights .

../../label_gpu /work/01255/siliu/CTR2017/benchmark_new/video03.mov 1
# label video_file n_threads
# Output: yolo.log which will be used as input for tracking.
# One task per node on skx, ibrun label_skx /work/01255/siliu/CTR2017/benchmark_new/video03.mov 47

To run track:
tracking video_name yolo.log
# two parameters: 1) the full path of video file 2) Yolo label output 

About output files:
yolo.log     -- Object recognition information obtained by the ¡°label¡± procedure, including
object class in Yolo, the bounding boxes information, and the confidence rate 
out_ncar.txt -- Number of card detected so far (accumulated)
out_ncar_Local.txt -- Number of cars detected in each frame
out_vector.txt -- The moving direction summary of each tracked object, including
the object rank, the first moving direction, the last moving direction, the inner 
and outer product of the two directions, and the concluded moving directions
out.log      -- Final result for each video file, including time step(frame) index, 
object index, moving flag, car flag, object class in Yolo, confidence rate, bounding 
box information, position displacement
out_bg.jpg  -- Background image generated from the input video
out.mov     -- Output video with all object recognition information


Documents about training your own model with YOLO. 
1) https://pjreddie.com/darknet/yolo/
2) https://github.com/AlexeyAB/darknet
Note: Use only one GPU when you start training for stability. 
https://github.com/AlexeyAB/Yolo_mark
https://github.com/unsky/yolo-for-windows-v2


