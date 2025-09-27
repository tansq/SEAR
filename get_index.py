from utils.dataloader import get_data

dataset = "coverage"
base_path, _, valid_file = get_data(dataset)
with open(dataset + ".txt", "w") as f:
    for idx, line in enumerate(valid_file):
        f.write(f"{idx},{line.split(' ')[1]}\n")
