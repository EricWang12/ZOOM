set -e

seed=1557241

default_attr=Young
default_exp=ffhq
default_gpu=0
default_classifier=full


attr=${1:-${default_attr}}
echo $attr

exp=${2:-${default_exp}}
echo $exp

gpu=${3:-${default_gpu}}
echo $gpu

token=$4
echo $token

classifier=${5:-${default_classifier}}
echo $classifier

# outlogdir=./log/${exp}-$(date +%F)
# mkdir -m 777 -p $outlogdir



mode="single"
step=0.2
iteration=100
bound=30

# mode="multiple"
# step=0.1
# iteration=100
# bound=30

# tokens for CLIP loss
token='["an Old face", "a Young face"]'

echo ${attr}-${mode}-${classifier}-${step}-${iteration}-${seed}-${bound}.txt
if test -f "${outdir}/${attr}-${mode}-${classifier}-${step}-${iteration}-${seed}-${bound}.txt"; then
    echo "log file exists."
    read -p "Do you wish to overwrite log file? " yn
    case $yn in
        [Yy]* ) :;;
        [Nn]* ) exit;;
        * ) echo "Please answer yes or no.";;
    esac
fi

# echo ${attr}-${mode}-${classifier}-${step}-${iteration}-${seed}-${bound}.txt
style_beta_or_channel=0.1
python zoom.py  --target_model_path="pretrained/victim_models/resnet50_${attr}_train${classifier}" \
                                    --seed=${seed} \
                                    --clip_token="${token}"\
                                    --attack_step_size=${step}\
                                    --style_beta_or_channel=${style_beta_or_channel}\
                                    --device="cuda:${gpu}" \
                                    --attack_iter=${iteration} \
                                    --attack_bound=${bound} \
                                    --mode=${mode} \
                                    --num_sample=1500 \
                                    --clip_weight=0.005 \
                                    --outdir=output/${exp}-$(date +%F)/${attr} \
                                    --experiment=${exp} 
                                    # --experiment=${exp} >  ${outlogdir}/${attr}-${mode}-${classifier}-${step}-${iteration}-${seed}-${bound}.txt
                                    

# python histogram.py -p "${outlogdir}/${attr}-${mode}-${classifier}-${step}-${iteration}-${bound}*"