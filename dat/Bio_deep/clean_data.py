import pandas as pd
from Bio import SeqIO
from pathlib import Path
from helpers import normalize_accession, filter_fasta_by_accessions

sequence_metadata = pd.read_csv("sequences.csv")
fasta_in = "sequences.fasta"
fasta_out = "filtered_sequences_outgroup.fasta"

# filter out sequences without Geo Location
sequence_metadata = sequence_metadata[sequence_metadata["Geo_Location"].notnull()]

# filter for countries with more than 15 sequences. 
counts = sequence_metadata["Geo_Location"].value_counts()
filtered = sequence_metadata[sequence_metadata["Geo_Location"].isin(counts[counts >= 15].index)]


filter_fasta_by_accessions(
    filtered, fasta_in, fasta_out,
    accession_col="Accession",
    include_also=None,
    ignore_version=False
)

filtered.to_csv("sequences_metadata_cleaned.csv")