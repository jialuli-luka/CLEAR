name=visual
flag="--attn soft --train visual
      --featdropout 0.3
      --angleFeatSize 128
      --feedback sample
      --mlWeight 0.2
      --feature_size 2048
      --subout max --dropout 0.5 --optim rms --lr 1e-4
      --objects_constraints objects_27.json
      --contrastive
      --con_weight 0.2
      --sim ce
      --iters 50000
      --batchSize 32
      --maxInput 160
      --maxAction 700"
mkdir -p snap/$name
CUDA_VISIBLE_DEVICES=$1 python r2r_src/train.py $flag --name $name
