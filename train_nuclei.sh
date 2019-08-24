python train.py --num_kernel 16\
                --kernel_size 3\
		        --lr 1e-3 \
		        --epoch 1\
			    --train_data  /home/mars/data/Nuclei/train.hdf5 \
			    --save_dir ./ \
                --device cuda\
                --dataset nuclei\
                --target_channels 4 \
                --optimizer adam\
                --model unet\
                --shuffle False \
                --num_workers 16 \
                --batch_size 32 \
                --epoch 200 \
                --gpu_ids 0\
                --experiment_name unet_nuclei_16