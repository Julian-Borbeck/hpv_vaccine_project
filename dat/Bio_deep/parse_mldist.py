import numpy as np
import pandas as pd

dist_file = "aligned.fasta.mldist"

with open(dist_file) as f:
    lines = [line.strip() for line in f if line.strip()]

# First line is number of taxa
n = int(lines[0])
names = []
data = []

for line in lines[1:]:
    parts = line.split()
    names.append(parts[0])
    row = list(map(float, parts[1:]))
    data.append(row)

df = pd.DataFrame(data, index=names, columns=names)

df.to_csv("aligned_dist.csv")