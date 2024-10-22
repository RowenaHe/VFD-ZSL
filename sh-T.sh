
python train-T.py --dataset AWA2 \
--ga 0.5 --nSample 5000 --gpu 2 --S_dim 1024 --D_dim 512 --R_dim 512 --out 2.0 \
--classifier_lr 0.003 --kl_warmup 0.01  --vae_dec_drop 0.5 --vae_enc_drop 0.1 \
--semantic_VAE_lr 0.0003 --ae_lr 0.000003 --gen_nepoch 220 \
--ae_drop 0.2 --evl_start 20000 --evl_interval 400 --manualSeed 6152 \
--classifier_steps 20 --part 0.3 --margin 0.1 \
--alter 2 --residual 0.1 --mi 0.1 --mi_epc 1
python train-T.py --dataset CUB  \
--ga 7 --nSample 1000 --gpu 2 --S_dim 1024 --D_dim 512 --R_dim 512 --dis 0.2  --out 2.0 \
--evl_start 0 --evl_interval 400 --gen_nepoch 600 \
--ae_lr 0.00005  --classifier_lr 0.0007 --semantic_VAE_lr 0.0003 \
--kl_warmup 0.001 --weight_decay 1e-8 \
--vae_enc_drop 0.3 --vae_dec_drop 0.3 --ae_drop 0.0 --margin 0.3 --classifier_steps 30 \
--alter 1 --residual 0.1 --mi 0.1  --mi_epc 3
python train-T.py --dataset SUN --ga 15 --dis 0.1 \
--nSample 400 --gpu 2 --S_dim 1024 --D_dim 512 --R_dim 512 --out 2.0 \
--ae_lr 0.00007 --semantic_VAE_lr 0.0003 --classifier_lr 0.005  \
--kl_warmup 0.001  --vae_dec_drop 0.5 --vae_enc_drop 0.4 --ae_drop 0.0 \
--manualSeed 6152 --classifier_steps 20 --part 0.3 --margin 0.1 \
--alter 1 --residual 0.1 --mi 0.1  --mi_epc 3 
python train-T.py --dataset FLO  \
--ga 2 --nSample 1200 --gpu 2 --S_dim 1024 --D_dim 512 --R_dim 512 --out 2.0 \
--classifier_lr 0.002 --kl_warmup 0.0003 \
--ae_lr 0.00003 --semantic_VAE_lr 0.0004 --gen_nepoch 400 \
--ae_drop 0.2 --vae_dec_drop 0.2 --vae_enc_drop 0.4 --evl_start 0 --evl_interval 400 --classifier_steps 20 \
--manualSeed 6152 --part 0.3 --margin 0.5 --residual 0.1 --mi 0.1 --mi_epc 10 --alter 2 
