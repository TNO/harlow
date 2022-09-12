def write_scores(writer, score_dict, iteration):
    # Writer log & various metric scores
    for k, v in score_dict.items():
        for i in range(len(v)):
            name = f"{k} surrogate {i}"
            writer.add_scalar(name, v[i], iteration)


def write_timer(writer, time_dict, iteration):
    for k, v in time_dict.items():
        writer.add_scalar(k, v, iteration)
