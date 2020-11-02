# tsne
# --sample_train_music_json ./Jsons/tsne.json \
#        --regression \

# Full : Untrack/evaluateResultRegTrip5600
#        --save_output_path ./Untrack/evaluateResultStyle \
#        --save_output_path Untrack/evaluateResultRegTrip5600 \
#        --save_output_path ./Untrack/evaluateResultClsTrip \
#        --regression \

        #--sample_train_paint_json ./Jsons/portrait/train_paint.json \
        #--load_model_path ./dum \

python3 evaluate.py \
        --load_model_path ./Source/last2.pth \
        --save_output_path ./Results \
        --sample_train_music_json ./Source/clips.json \
        --gpu_ids 0 \
        --paint_resize_min_edge 96 \
        --paint_crop_H 64 \
        --paint_crop_W 64 \
        --audio_length 8.91 \
        --z_dim 256 \
        --mean 1847.096 \
        --std 103.065 \
        --year_base 1480 \
        --eva_base $1 \
        --eva_count $2 \
        --regression \
        --split_num 3
