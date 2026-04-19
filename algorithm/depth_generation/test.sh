#CUDA_VISIBLE_DEVICES=4 python test_simple_3m.py \
#--image_path /mnt/haze/data/source_test/  \
#--save_path /mnt/haze/data/result_test/  \
#--model_name mono+stereo_640x192    \
#--add_fog_method mean+min+max
#2>&1 | tee mylog.log

python test_simple_3m.py --image_path ./assets/test_image.jpg --save_path ./result_test/ --model_name mono+stereo_640x192 --add_fog_method mean+min+max
2>&1 | tee mylog.log
