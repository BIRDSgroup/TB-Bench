#!/usr/bin/env python3
import os
import gzip
import shutil
import subprocess
import tempfile
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
import glob
import re
import multiprocessing
import argparse
import sys

# --- Parameters ---
# IMPORTANT: Update these paths to match your server/cluster layout before running.
vcf_source_dir = "public-data/test/Sid_test_vcf/"
min_size_bytes = 1 * 1024  # Minimum valid VCF.gz size in bytes (1 KB); smaller files are skipped.

# Absolute path to the Perl SNP-concatenation script.
# perl_dir is derived from this path and is used as the working directory for the Perl process.
perl_script = "snpConcatenater_w_exclusion_frompilonvcf_2.9.pl"
perl_dir = os.path.dirname(perl_script)
unzipped_dir = os.path.join(perl_dir, "unzipped_vcf")
os.makedirs(unzipped_dir, exist_ok=True)

# Directory holding one reference FASTA per locus (used as MAFFT profile references).
reference_dir = "test_vcf_prep/fasta_files"
ref_files = glob.glob(os.path.join(reference_dir, "*.fasta"))

# Genome info
genome_start = 0
genome_end = 4411532  # H37Rv genome length

# --- Loci definitions ---
loci_regions = {
    # Existing loci
    "acpM-kasA": (2517695, 2519365),
    "gid": (4407528, 4408334),
    "rpsA": (1833378, 1834987),
    "clpC": (4036731, 4040937),
    "embCAB": (4239663, 4249810),
    "aftB-ubiA": (4266953, 4269833),
    "rrs-rrl": (1471576, 1477013),
    "ethAR": (4326004, 4328199),
    "oxyR-ahpC": (2725477, 2726780),
    "tlyA": (1917755, 1918746),
    "KatG": (2153235, 2156706),
    "rpsL": (781311, 781934),
    "rpoBC": (759609, 767320),
    "FabG1-inhA": (1672457, 1675011),
    "eis": (2713783, 2716314),
    "gyrBA": (4997, 9818),
    "panD": (4043041, 4045210),
    "pncA": (2287883, 2289599),
    # New loci
    "alr": (3840194, 3841420),
    "ald": (3086820, 3087935),
    "ddlA": (3336796, 3337917),
    "cycA": (1929786, 1931456),
    "thyA": (3073680, 3074471),
    "mmpL5": (775586, 778480),
    "mmpS5": (778477, 778905),
    "Rv0678": (778990, 779487),
    "atpE": (1461045, 1461290),
    "pepQ": (2859300, 2860418),
    "rplC": (800809, 801462),
    "rrl": (1473658, 1476795),
}

# --- Helpers ---
def find_vcf_path(sample_id):
    """Find .vcf.gz file for a sample_id in vcf_source_dir."""
    pattern = os.path.join(vcf_source_dir, f"{sample_id}*.vcf.gz")
    matches = glob.glob(pattern)
    return matches[0] if matches else None

def is_valid_gz(vcf_gz_path):
    """Return True if gz file exists and is larger than min_size_bytes."""
    return vcf_gz_path is not None and os.path.getsize(vcf_gz_path) > min_size_bytes

def find_matching_ref(locus):
    """Find a fasta in reference_dir that matches locus name (case-insensitive substring)."""
    pattern = re.compile(re.escape(locus), re.IGNORECASE)
    for ref_file in ref_files:
        if pattern.search(os.path.basename(ref_file)):
            return ref_file
    return None

def align_locus(drug, locus, output_dir, aligned_output_dir, mafft_threads):
    new_fasta = os.path.join(output_dir, f"{locus}.fasta")
    aligned_out = os.path.join(aligned_output_dir, f"{locus}_aligned.fasta")
    ref_path = find_matching_ref(locus)

    if ref_path and os.path.exists(new_fasta) and os.path.getsize(new_fasta) > 0:
        try:
            with open(aligned_out, "w") as out_aln:
                subprocess.run(
                    ["mafft", "--thread", str(mafft_threads),
                     "--add", new_fasta, "--keeplength", ref_path],
                    stdout=out_aln,
                    stderr=subprocess.DEVNULL,
                    check=True
                )
            return f"{drug}:{locus}: ALIGNED -> {aligned_out}"
        except subprocess.CalledProcessError as e:
            return f"{drug}:{locus}: ERROR -> mafft failed ({e})"
        except Exception as e:
            return f"{drug}:{locus}: ERROR -> {str(e)}"
    else:
        reason = []
        if not ref_path:
            reason.append("missing reference")
        if not os.path.exists(new_fasta) or os.path.getsize(new_fasta) == 0:
            reason.append("empty or missing input FASTA")
        return f"{drug}:{locus}: SKIPPED ({', '.join(reason)})"

def process_drug(drug, meta_file, mafft_threads, unzipped_map, skip_alignment=False):
    """
    Process all samples for a single anti-TB drug:
      1. Read the metadata file (sample_id → R/S label).
      2. Look up each sample's pre-unzipped VCF path from `unzipped_map`.
      3. For every locus in `loci_regions`, build a BED exclusion file for
         the complement of the locus, then call the Perl snpConcatenater script
         to produce a per-locus FASTA containing one entry per sample.
      4. Optionally align each per-locus FASTA against the H37Rv reference
         using MAFFT (`--add --keeplength`).

    Parameters
    ----------
    drug : str
        Short drug code (e.g. 'LFX').
    meta_file : str
        Path to the metadata TSV: two columns — sample_id and R/S/I label.
    mafft_threads : int
        Number of CPU threads to pass to each MAFFT invocation.
    unzipped_map : dict
        Mapping of sample_id -> absolute path to the already-unzipped VCF file.
    skip_alignment : bool, optional
        If True, skip the MAFFT alignment step (Perl FASTAs only). Default False.

    Returns
    -------
    str
        A status message summarising outcomes for this drug.
    """
    print(f"\n=== Processing {drug} ===")

    if not os.path.exists(meta_file):
        return f"[WARNING] No metadata file for {drug} at {meta_file}, skipping."

    try:
        df = pd.read_csv(meta_file, sep=r"\s+", header=None, names=["Sample_id", drug], engine="python")
    except Exception as e:
        return f"[ERROR] Could not read metadata for {drug}: {e}"

    # Attach the on-disk unzipped VCF path; rows without a matching path are dropped.
    df["unzipped_path"] = df["Sample_id"].map(unzipped_map)
    filtered_df = df[df["unzipped_path"].notna()]

    if filtered_df.empty:
        return f"[WARNING] No valid/unzipped samples for {drug}, skipping."

    output_dir = os.path.join(perl_dir, f"perl_output_fastas/{drug}")
    aligned_output_dir = os.path.join(perl_dir, f"fasta_files_aligned_final_20_not_in_master/{drug}")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(aligned_output_dir, exist_ok=True)

    # Build a per-drug VCF list file consumed by the Perl script.
    input_list_file = os.path.join(perl_dir, f"input_vcf_{drug}.txt")

    unzipped_files = []
    for _, row in filtered_df.iterrows():
        u_path = os.path.abspath(row["unzipped_path"])
        unzipped_files.append(u_path)

    with open(input_list_file, "w") as f:
        for vcf in unzipped_files:
            f.write(f"{vcf}\n")

    # The Perl script always reads from a fixed filename in its working directory;
    # copy the drug-specific list to that expected location.
    default_input_vcf = os.path.join(perl_dir, "input_vcf_files.txt")
    shutil.copy(input_list_file, default_input_vcf)

    idfail_tab = "IDfail.tab"  # Empty placeholder; list QC-failed sample IDs here to exclude them.

    for locus, (start, end) in loci_regions.items():
        region = f"{start}-{end}"
        output_fasta = os.path.join(output_dir, f"{locus}.fasta")

        # Build a BED file that covers everything OUTSIDE the target locus.
        # The Perl script uses this exclusion BED to skip variants outside the
        # region of interest, producing a whole-region MSA for the locus.
        with tempfile.NamedTemporaryFile(mode='w', suffix=".BED", delete=False) as tmp_bed:
            if start > genome_start:
                tmp_bed.write(f"NC_000962.3\t{genome_start}\t{start - 1}\n")
            if end < genome_end:
                tmp_bed.write(f"NC_000962.3\t{end + 1}\t{genome_end}\n")
            exclude_bed_path = tmp_bed.name

        command = [
            "perl", perl_script,
            exclude_bed_path,
            idfail_tab,
            "INDEL",
            "REGION",
            region,
            "pos"
        ]
        try:
            with open(output_fasta, "w") as out_f:
                subprocess.run(command, stdout=out_f, stderr=subprocess.DEVNULL, check=True)
            print(f"{drug}: Created FASTA for locus {locus} -> {output_fasta}")
        except subprocess.CalledProcessError:
            print(f"{drug}: perl script failed for locus {locus} (region {region}). Output file may be incomplete.")
        except Exception as e:
            print(f"{drug}: ERROR running perl for locus {locus}: {e}")

        try:
            os.remove(exclude_bed_path)
        except OSError:
            pass

    # Check if alignment should be skipped
    if skip_alignment:
        return f"{drug}: Perl processing complete (alignment skipped). FASTAs in {output_dir}"

    log_file = os.path.join(aligned_output_dir, f"{drug}_alignment_log.txt")
    with open(log_file, "w") as log:
        for locus in loci_regions:
            result = align_locus(drug, locus, output_dir, aligned_output_dir, mafft_threads)
            print(result)
            log.write(result + "\n")

    return f"{drug}: complete. Log -> {log_file}"

def global_unzip_all_samples(drug_meta_map, unzipped_dir, min_size_bytes):
    """
    Read all metadata files, collect unique sample IDs, unzip each sample_id.vcf.gz
    into unzipped_dir/sample_id.vcf (no drug prefix). Returns a dict sample_id -> unzipped_path.
    """
    sample_ids = set()
    for drug, meta_path in drug_meta_map.items():
        if not os.path.exists(meta_path):
            print(f"[global_unzip] Warning: metadata file not found for {drug}: {meta_path}")
            continue
        try:
            df = pd.read_csv(meta_path, sep=r"\s+", header=None, names=["Sample_id", drug], engine="python")
            sample_ids.update(df["Sample_id"].astype(str).tolist())
        except Exception as e:
            print(f"[global_unzip] Warning: could not read {meta_path}: {e}")
            continue

    print(f"[global_unzip] Found {len(sample_ids)} unique sample IDs across metadata files.")

    unzipped_map = {}
    for sid in sample_ids:
        gz_path = find_vcf_path(sid)
        if not gz_path:
            print(f"[global_unzip] No gz for sample {sid}, skipping.")
            continue
        try:
            if not is_valid_gz(gz_path):
                print(f"[global_unzip] gz too small or invalid for sample {sid}: {gz_path}")
                continue
        except Exception as e:
            print(f"[global_unzip] Could not stat {gz_path} for sample {sid}: {e}")
            continue

        out_path = os.path.join(unzipped_dir, f"{sid}.vcf")
        if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
            unzipped_map[sid] = os.path.abspath(out_path)
            print(f"[global_unzip] Already unzipped: {sid} -> {out_path}")
            continue

        try:
            with gzip.open(gz_path, 'rb') as f_in, open(out_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
            unzipped_map[sid] = os.path.abspath(out_path)
            print(f"[global_unzip] Unzipped {gz_path} -> {out_path}")
        except Exception as e:
            print(f"[global_unzip] Failed to unzip {gz_path} for sample {sid}: {e}")

    print(f"[global_unzip] Completed. Successfully unzipped {len(unzipped_map)} samples.")
    return unzipped_map

def parse_drug_metadata(args):
    """
    Parse drug metadata from various input methods.
    Returns a dictionary {drug_name: metadata_path}
    """
    drug_meta_map = {}
    
    # Method 1: Individual drug arguments (--LFX, --AMK, etc.)
    for drug in ['LFX', 'AMK', 'KAN', 'CAP', 'MFX', 'OFX', 'ETO', 'CIP', 
                 'CYC', 'MB', 'PTO', 'PAS', 'BDQ', 'LZD']:
        value = getattr(args, drug, None)
        if value:
            drug_meta_map[drug] = value
    
    # Method 2: Directory containing all metadata files (--meta-dir)
    if args.meta_dir:
        if not os.path.isdir(args.meta_dir):
            print(f"Error: --meta-dir path is not a directory: {args.meta_dir}")
            sys.exit(1)
        
        for drug in ['LFX', 'AMK', 'KAN', 'CAP', 'MFX', 'OFX', 'ETO', 'CIP',
                     'CYC', 'MB', 'PTO', 'PAS', 'BDQ', 'LZD']:
            meta_file = os.path.join(args.meta_dir, drug, f"{drug}_metadata.txt")
            if os.path.exists(meta_file) and drug not in drug_meta_map:
                drug_meta_map[drug] = meta_file
    
    # Method 3: Config file (--config)
    if args.config:
        if not os.path.exists(args.config):
            print(f"Error: Config file not found: {args.config}")
            sys.exit(1)
        
        try:
            with open(args.config, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    parts = line.split('=', 1)
                    if len(parts) == 2:
                        drug, path = parts[0].strip(), parts[1].strip()
                        if drug not in drug_meta_map:
                            drug_meta_map[drug] = path
        except Exception as e:
            print(f"Error reading config file: {e}")
            sys.exit(1)
    
    if not drug_meta_map:
        print("Error: No drug metadata files specified. Use --help for usage information.")
        sys.exit(1)
    
    return drug_meta_map

def main():
    parser = argparse.ArgumentParser(
        description='Process VCF files for multiple TB drug resistance analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Method 1: Individual drug arguments
  %(prog)s --LFX /path/to/LFX_metadata.txt --AMK /path/to/AMK_metadata.txt
  
  # Method 2: Directory containing drug subdirectories
  %(prog)s --meta-dir /data/public-data/TB_data_alldrugs
  
  # Method 3: Config file
  %(prog)s --config drugs.conf
  
  # Mix methods
  %(prog)s --meta-dir /data/public-data/TB_data_alldrugs --LFX /custom/path/LFX_metadata.txt
  
  # Run only Perl script without alignment
  %(prog)s --meta-dir /data/public-data/TB_data_alldrugs --perl-only

Config file format (drugs.conf):
  LFX=/path/to/LFX_metadata.txt
  AMK=/path/to/AMK_metadata.txt
  # Lines starting with # are comments
        '''
    )
    
    # Method 1: Individual drug arguments
    drug_group = parser.add_argument_group('Individual drug metadata files')
    for drug in ['LFX', 'AMK', 'KAN', 'CAP', 'MFX', 'OFX', 'ETO', 'CIP',
                 'CYC', 'MB', 'PTO', 'PAS', 'BDQ', 'LZD']:
        drug_group.add_argument(f'--{drug}', metavar='PATH',
                               help=f'{drug} metadata file path')
    
    # Method 2: Directory-based input
    parser.add_argument('--meta-dir', metavar='DIR',
                       help='Base directory containing drug subdirectories (e.g., /data/public-data/TB_data_alldrugs)')
    
    # Method 3: Config file
    parser.add_argument('--config', metavar='FILE',
                       help='Config file with drug=path pairs (one per line)')
    
    # Additional options
    parser.add_argument('--max-workers', type=int, metavar='N',
                       help='Maximum parallel workers (default: auto-detect based on CPU cores)')
    
    parser.add_argument('--perl-only', action='store_true',
                       help='Run only Perl script to generate FASTAs, skip MAFFT alignment')
    
    parser.add_argument('--list-drugs', action='store_true',
                       help='List configured drugs and exit')
    
    args = parser.parse_args()
    
    # Parse drug metadata from various sources
    drug_metadata_files = parse_drug_metadata(args)
    
    if args.list_drugs:
        print("Configured drugs:")
        for drug, path in sorted(drug_metadata_files.items()):
            exists = "?" if os.path.exists(path) else "?"
            print(f"  {exists} {drug:6s} -> {path}")
        sys.exit(0)
    
    print(f"Processing {len(drug_metadata_files)} drugs: {', '.join(sorted(drug_metadata_files.keys()))}")
    
    if args.perl_only:
        print("*** Running in Perl-only mode (MAFFT alignment will be skipped) ***")
    
    # Pre-unzip all sample files
    unzipped_map = global_unzip_all_samples(drug_metadata_files, unzipped_dir, min_size_bytes)
    
    # Determine parallelization
    total_cores = multiprocessing.cpu_count()
    drugs = list(drug_metadata_files.items())
    
    threads_per_mafft = max(1, total_cores // max(1, len(drugs)))
    
    if args.max_workers:
        max_workers = min(args.max_workers, len(drugs))
    else:
        max_workers = min(len(drugs), max(1, total_cores // threads_per_mafft))
    
    print(f"Detected {total_cores} cores. Running up to {max_workers} drugs in parallel, "
          f"{threads_per_mafft} threads per MAFFT job.")
    
    # Submit per-drug jobs
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(process_drug, drug, meta_file, threads_per_mafft, unzipped_map, args.perl_only): drug
            for drug, meta_file in drugs
        }
        for future in as_completed(futures):
            try:
                print(future.result())
            except Exception as e:
                print(f"[Main] A drug worker failed: {e}")
    
    print("\n=== All drugs processed ===")

if __name__ == "__main__":
    main()