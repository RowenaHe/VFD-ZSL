
python train-H.py --dataset FLO --dis 0.1 \
--ga 2 --nSample 1200 --gpu 4 --S_dim 1024 --D_dim 512 --R_dim 512 --out 2.0 \
--classifier_lr 0.002 --kl_warmup 0.0003 \
--ae_lr 0.00003 --semantic_VAE_lr 0.0004 --gen_nepoch 400 \
--ae_drop 0.2 --vae_dec_drop 0.2 --vae_enc_drop 0.4 --evl_start 0 --evl_interval 400 --classifier_steps 20 \
--manualSeed 6152 --part 0.3 --margin 0.2 --residual 0.1 --mi 0.1 --mi_epc 2 --alter 2
# 43.54
python train-H.py --dataset SUN --ga 30 --dis 0.5 \
--nSample 400 --gpu 4 --S_dim 1024 --D_dim 512 --R_dim 512  --out 2.0 \
--ae_lr 0.00006 --semantic_VAE_lr 0.0003 --classifier_lr 0.005  \
--kl_warmup 0.001 --vae_enc_drop 0.4 --vae_dec_drop 0.2 --ae_drop 0.4 \
--manualSeed 6152 --classifier_steps 20 --part 0.3 --margin 0.1 \
--alter 1 --residual 0.1 --mi 0.1  --mi_epc 3
# 56.88
python train-H.py --dataset CUB  \
--ga 7 --nSample 1000 --gpu 4 --S_dim 1024 --D_dim 512 --R_dim 512 --dis 0.3  --out 2.0 \
--evl_start 0 --evl_interval 400 --gen_nepoch 600 \
--ae_lr 0.00004  --classifier_lr 0.002 --semantic_VAE_lr 0.0006 \
--kl_warmup 0.001 --weight_decay 1e-8 \
--vae_enc_drop 0.3 --vae_dec_drop 0.3 --ae_drop 0.0 --margin 0.3 --classifier_steps 30 \
--alter 1 --residual 0.05 --mi 0.06  --mi_epc 3
# 68.13
python train-H.py --dataset AWA2 --dis 0.1 \
--ga 0.5 --nSample 5000 --gpu 4 --S_dim 1024 --D_dim 512 --R_dim 512 --out 2.0 \
--classifier_lr 0.001 --kl_warmup 0.01  --vae_dec_drop 0.5 --vae_enc_drop 0.4 \
--semantic_VAE_lr 0.0006 --ae_lr 0.000006 --gen_nepoch 220 \
--ae_drop 0.2 --evl_start 20000 --evl_interval 400 --manualSeed 6152 \
--classifier_steps 20 --part 0.3 --margin 0.1 \
--alter 1 --residual 0.1 --mi 0.1 --mi_epc 2

