#set "BERT4rec_HOME_DIR" as env variable in your shell!!!!!
BERT4rec_HOME_DIR=${BERT4rec_HOME_DIR}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}
dataset_name="rsc15"
max_predictions_per_seq=10
masked_lm_prob=0.5

dim=100
num_train_steps=200

mask_prob=1.0
prop_sliding_window=0.1
dupe_factor=10

for batch_size in 8 16 32 64; do
  for max_seq_length in 10 20 30; do
    for index in {1..5}; do
      signature="-mp${mask_prob}-sw${prop_sliding_window}-mlp${masked_lm_prob}-df${dupe_factor}-mpps${max_predictions_per_seq}"
      additional_props="msl${max_seq_length}-test_split${index}-batch_size${batch_size}"
      python -u run.py \
      --train_input_file=./data/${dataset_name}/train/max_seq_length-${max_seq_length}/${dataset_name}${signature}_split-${index}.train.tfrecord \
      --test_input_file=./data/${dataset_name}/train/max_seq_length-${max_seq_length}/${dataset_name}${signature}_split-${index}.test.tfrecord \
      --vocab_filename=./data/${dataset_name}/train/max_seq_length-${max_seq_length}/${dataset_name}${signature}_split-${index}.vocab \
      --user_history_filename=./data/${dataset_name}/train/max_seq_length-${max_seq_length}/${dataset_name}${signature}_split-${index}.his \
      --checkpointDir=${BERT4rec_HOME_DIR}/results/hyper_opt/${dataset_name}/${additional_props}___ \
      --signature=${signature}-${dim} \
      --do_train=True \
      --do_eval=True \
      --bert_config_file=./bert_train/bert_config_${dataset_name}_msl-${max_seq_length}_${dim}.json \
      --batch_size=${batch_size} \
      --max_seq_length=${max_seq_length} \
      --max_predictions_per_seq=${max_predictions_per_seq} \
      --num_train_steps=${num_train_steps} \
      --num_warmup_steps=100 \
      --learning_rate=1e-4 \
      --eval_split_output=./data/${dataset_name}/train/max_seq_length-${max_seq_length}/batch_size${batch_size}_eval.txt
    done
  done
done

