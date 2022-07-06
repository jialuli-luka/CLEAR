name=agent
flag="--attn soft --train listener
      --featdropout 0.3
      --angleFeatSize 128
      --feedback sample
      --mlWeight 0.4
      --feature_size 2048
      --subout max --dropout 0.5 --optim rms --lr 1e-4
      --dataset RxR
      --load_encoder snap/encoder/state_dict/best_val_unseen_loss
      --load_visual snap/visual/state_dict/best_val_unseen_loss
      --accumulateGrad
      --iters 1600000
      --batchSize 6
      --maxInput 160
      --maxAction 70"
mkdir -p snap/$name
CUDA_VISIBLE_DEVICES=$1 python r2r_src/train.py $flag --name $name

# Try this with file logging:
# CUDA_VISIBLE_DEVICES=$1 unbuffer python r2r_src/train.py $flag --name $name | tee snap/$name/log
