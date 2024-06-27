torchrun --nproc_per_node 4 -m training.main --batch-size=32 --lr=1e-5 --wd=0.1 --epochs=100 --workers=4 \
--model ViT-B-16 --pretrained openai --warmup 1000  --zeroshot-frequency 10 --dataset-type coco_caption  \
--test-type coco_panoptic --train-data data/coco/wusize/captions_train2017_tags_allcaps.json \
--val-data data/coco/annotations/panoptic_val2017.json \
--embed-path metadata/coco_panoptic_clip_hand_craft_ViTB16.npy --train-image-root="data/coco/train2017" \
--val-image-root data/coco/val2017  --cache-dir checkpoints --log-every-n-steps 50 \
--lock-image --save-frequency 100 --lock-image-unlocked-groups 6 --extract-type="v2" \
--name clim_coco_100_save100_test10_openai_vitb16_6layers --downsample-factor 16 --det-image-size 1024 \
--alpha 0.7 --train-image-size 1024
