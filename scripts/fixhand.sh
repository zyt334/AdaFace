#!/usr/bin/fish
python3 main.py --base configs/stable-diffusion/v1-finetune-ada.yaml -t --actual_resume models/stable-diffusion-v-1-5/v1-5-dste.ckpt --gpus 1, --data_root data-extra/fixhand/ -n fixhand-ada --no-test --max_steps 2500 --placeholder_string z --init_words hand --init_word_weights 1 --broad_class 1 --randomize_clip_skip_weights --num_vectors_per_token 9 --use_conv_attn --clip_last_layers_skip_weights 1 2 2 --cls_delta_token hand --lr 3e-4 --compos_placeholder_prefix "man with, woman with, boy with, girl with" --ada_embedding_reg_weight 1e-2 --layerwise_lora_rank 5