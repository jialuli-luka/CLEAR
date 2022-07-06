name=encoder
flag="--attn soft --train mbert
      --featdropout 0.3
      --angleFeatSize 128
      --feedback sample
      --mlWeight 0.2
      --feature_size 512
      --subout max --dropout 0.5 --optim rms --lr 1e-4
      --con_weight 0.2
      --accumulateGrad
      --iters 100000
      --batchSize 8
      --maxInput 160
      --maxAction 70"
mkdir -p snap/$name
CUDA_VISIBLE_DEVICES=$1 python r2r_src/train.py $flag --name $name
