import subprocess
from Bio import AlignIO
from Bio.Phylo.TreeConstruction import DistanceCalculator
import numpy as np

input_fasta = "filtered_sequences_outgroup.fasta"          # input protein fasta
aligned_fasta = "aligned.fasta"      # MUSCLE output
muscle_bin = "muscle"              # path to MUSCLE binary if not in PATH

cmd = [muscle_bin, "-align", input_fasta, "-output", aligned_fasta] #assumes muscle5

subprocess.run(cmd, check=True)

alignment = AlignIO.read(aligned_fasta, "fasta")

print(f"Loaded alignment with {len(alignment)} sequences of length {alignment.get_alignment_length()}")

cmd2 = ["iqtree", "-s", aligned_fasta, "-m" ,"BLOSUM62" ,"-keep-ident", "-safe"] #assumes iqtree3

subprocess.run(cmd2, check=True)