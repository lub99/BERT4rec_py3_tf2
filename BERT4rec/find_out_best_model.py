if __name__ == '__main__':
    dataset_name = "rsc15"
    max_seq_lengths = [10, 20]
    batch_sizes = [8]

    hyper_param_dir = "./data/" + dataset_name + "/train/"
    max_avg_ndcg = -1
    optim_batch_size = -1
    optim_max_seq_length = -1
    for msl in max_seq_lengths:
        for batch in batch_sizes:
            result_filename = "batch_size" + str(batch) + "_eval.txt"
            with open(hyper_param_dir + "max_seq_length-" + str(msl) + "/" + result_filename, "r") as f:
                line_num = 0
                sum_of_ndcg = 0
                for line in f:
                    line_num += 1
                    sum_of_ndcg += float(line)
                avg_ndcg = sum_of_ndcg / line_num

                if avg_ndcg > max_avg_ndcg:
                    max_avg_ndcg = avg_ndcg
                    optim_batch_size = batch
                    optim_max_seq_length = msl

    print("Best model have average NDCG@20={}.\nOptimal parameters are: batch_size={}, max_seq_length={}"
          .format(max_avg_ndcg,
                  optim_batch_size,
                  optim_max_seq_length))
