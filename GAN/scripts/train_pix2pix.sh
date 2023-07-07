python train.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --which_model_netG unet_256 --which_direction BtoA --lambda_A 100 --dataset_mode aligned --no_lsgan --norm batch

python GAN/train.py --dataroot data_preprocessing/tsinghuaDaimlerDataset/images_aligned_crop --name trials --model pix2pix --which_model_netG unet_256 --which_direction BtoA --lambda_A 100 --dataset_mode aligned --use_spp --no_lsgan --norm batch --gpu_ids -1 --display_id 0 --niter 5 --niter_decay 5 --loadSize 542 --fineSize 512
python GAN/train.py --dataroot data_preprocessing/tsinghuaDaimlerDataset/images_aligned_one --name one --model pix2pix --which_model_netG unet_256 --which_direction BtoA --lambda_A 100 --dataset_mode aligned --use_spp --no_lsgan --norm batch --gpu_ids -1 --display_id 0

python GAN/datasets/combine_A_and_B.py --fold_A data_preprocessing/tsinghuaDaimlerDataset/images_full_crop/test  --fold_B data_preprocessing/tsinghuaDaimlerDataset/images_noise_crop/test --fold_AB data_preprocessing/tsinghuaDaimlerDataset/images_aligned_crop/images/test --num_imgs 504
