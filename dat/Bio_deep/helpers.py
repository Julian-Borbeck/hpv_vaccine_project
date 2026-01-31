from Bio import SeqIO
from pathlib import Path

# normalize uniprot accession identifiers
def normalize_accession(x, ignore_version = True):
    tok = str(x).strip().split()[0]
    if '|' in tok:
        tok = tok.split('|')[1]
    if ignore_version and '.' in tok:
        tok = tok.split('.', 1)[0]
    return tok

# filter and export
def filter_fasta_by_accessions(
    df,
    fasta_in,
    fasta_out,
    accession_col = "Accession",
    include_also = None,
    ignore_version = True
):

    acc2len = {}
    
    include_also = include_also or set()

    # Build a set of normalized accessions from the dataframe
    df_accs_raw = (
        df[accession_col]
        .dropna()
        .astype(str)
        .map(lambda s: s.strip())
    )
    df_accs_norm = {normalize_accession(a, ignore_version=ignore_version) for a in df_accs_raw}

    # Always include extras like the outgroup
    extra_norm = {normalize_accession(a, ignore_version=ignore_version) for a in include_also}
    wanted = df_accs_norm | extra_norm

    kept_records = []
    seen_norm_ids = set()

    for rec in SeqIO.parse(fasta_in, "fasta"):
        rec_norm = normalize_accession(rec.id, ignore_version=ignore_version)
        # If header id didn’t match, try the name/description too
        if rec_norm in wanted:

            acc2len[rec_norm] = len(str(rec.seq))
            
            kept_records.append(rec)
            seen_norm_ids.add(rec_norm)
        else:
            # sometimes the accession is only in the description
            desc_norm = normalize_accession(rec.description, ignore_version=ignore_version)
            if desc_norm in wanted:
                kept_records.append(rec)
                seen_norm_ids.add(desc_norm)

    # Write the filtered FASTA
    Path(fasta_out).parent.mkdir(parents=True, exist_ok=True)
    SeqIO.write(kept_records, fasta_out, "fasta")

    # Simple report
    missing = sorted(wanted - seen_norm_ids)
    print(f"Input FASTA: {fasta_in}")
    print(f"Output FASTA: {fasta_out}")
    print(f"Sequences kept: {len(kept_records)}")
    print(f"Unique accessions requested: {len(wanted)}")
    if missing:
        print(f"Accessions not found in FASTA ({len(missing)}):")
        print(", ".join(missing[:20]) + (" ..." if len(missing) > 20 else ""))