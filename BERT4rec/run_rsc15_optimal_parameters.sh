#set "BERT4rec_HOME_DIR" as env variable in your shell!!!!!
BERT4rec_HOME_DIR=${BERT4rec_HOME_DIR}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}
train_dataset_name="rsc15-clicks_train_full.4"
test_dataset_name="rsc15-clicks_test.4"
max_seq_length=10
max_predictions_per_seq=10
#max_predictions_per_seq=30
masked_lm_prob=0.5
#masked_lm_prob=0.6

dim=100
batch_size=16
num_train_steps=400000

mask_prob=1.0
prop_sliding_window=0.1
dupe_factor=10
pool_size=10

signature="-mp${mask_prob}-sw${prop_sliding_window}-mlp${masked_lm_prob}-df${dupe_factor}-mpps${max_predictions_per_seq}-msl${max_seq_length}"

train_input_file=./data/rsc15/final/${train_dataset_name}${signature}.train.tfrecord
if [ -f "$train_input_file" ]
then
  echo "$train_input_file file already exist. No need for dataset construction!"
else
  { python -u my_gen_data_fin.py \
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
      --input_data_dir=./data/rsc15/input/ \
      --output_data_dir=./data/rsc15/final/
      }
fi


python -u run.py \
    --train_input_file=./data/rsc15/final/${train_dataset_name}${signature}.train.tfrecord \
    --test_input_file=./data/rsc15/final/${train_dataset_name}${signature}.test.tfrecord \
    --vocab_filename=./data/rsc15/final/${train_dataset_name}${signature}.vocab \
    --user_history_filename=./data/rsc15/final/${train_dataset_name}${signature}.his \
    --checkpointDir=${BERT4rec_HOME_DIR}/results/${train_dataset_name} \
    --signature=${signature}-${dim} \
    --do_train=True \
    --do_eval=True \
    --bert_config_file=./bert_train/bert_config_rsc15_msl-10_100.json \
    --batch_size=${batch_size} \
    --max_seq_length=${max_seq_length} \
    --max_predictions_per_seq=${max_predictions_per_seq} \
    --num_train_steps=${num_train_steps} \
    --num_warmup_steps=100 \
    --learning_rate=1e-4 \
    --test_results_output=./data/rsc15/final/result.txt \
    --train_input_txt=./data/rsc15/input/rsc15-clicks_train_full.txt
