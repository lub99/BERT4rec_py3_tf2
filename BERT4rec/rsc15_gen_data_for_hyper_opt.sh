#set "BERT4rec_HOME_DIR" as env variable in your shell!!!!!
BERT4rec_HOME_DIR=${BERT4rec_HOME_DIR}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}
dataset_name="rsc15"
train_dataset_name="rsc15-clicks_train_full.4"
test_dataset_name="rsc15-clicks_test.4"
max_predictions_per_seq=10
masked_lm_prob=0.5


mask_prob=1.0
prop_sliding_window=0.1
dupe_factor=10
pool_size=20
k_fold=5

signature="-mp${mask_prob}-sw${prop_sliding_window}-mlp${masked_lm_prob}-df${dupe_factor}-mpps${max_predictions_per_seq}"


for max_seq_length in 10 30 50; do
    train_input_file=./data/${dataset_name}/train/max_seq_length-${max_seq_length}/${dataset_name}${signature}_split-5.train.tfrecord
    if [ -f "$train_input_file" ]
    then
      echo "$train_input_file file already exist. No need for dataset construction!"
    else
      { python -u hyper_opt_gen_data_fin.py \
          --train_dataset_name=${train_dataset_name} \
          --test_dataset_name=${test_dataset_name} \
          --max_seq_length=${max_seq_length} \
          --max_predictions_per_seq=${max_predictions_per_seq} \
          --mask_prob=${mask_prob} \
          --dupe_factor=${dupe_factor} \
          --masked_lm_prob=${masked_lm_prob} \
          --prop_sliding_window=${prop_sliding_window} \
          --signature=${signature} \
          --pool_size=${pool_size} \
          --dataset_name=${dataset_name} \
          --k_fold=${k_fold}
          }
    fi
done
