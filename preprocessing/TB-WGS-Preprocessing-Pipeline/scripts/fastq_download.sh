#!/bin/bash

###############################################################################
# Whole Genome Sequencing (WGS) processing pipeline for MTB isolates
#
# For each sample:
#   1. Download raw reads from NCBI SRA
#   2. Convert SRA → paired-end FASTQ
#   3. Merge multiple runs (if present)
#   4. Quality control & trimming (fastp)
#   5. Align reads to MTB H37Rv reference (bwa)
#   6. Sort, index, and QC alignments (samtools)
#   7. Remove PCR duplicates (Picard)
#   8. Call variants (FreeBayes)
#   9. Compute coverage statistics
#
# Input (per line in resolved_1720.tsv):
#   Column 1: Sample ID
#   Column 2: Comma-separated SRA run IDs
#
# Output (per sample directory):
#   - BAM files
#   - VCF (compressed)
#   - Depth & coverage statistics
###############################################################################

# Suppress Picard JMX warnings
export _JAVA_OPTIONS="-Djava.util.logging.config.file=/dev/null"

process_sample_new() {

    ###########################################################################
    # Parse input arguments
    ###########################################################################
    sam_id=${1}   # Sample identifier (used as directory and file prefix)
    sra_id=${2}   # One or more SRA run IDs (comma-separated)

    # Define sample-specific output directory
    outdir="./${sam_id}"

    # Create sample-specific log file in logs folder
    logfile="../logs/${sam_id}.log"
    mkdir -p ../logs

    # Print expected VCF output path (useful for logging/debugging)
    echo "$outdir/${sam_id}_n.vcf.gz"

    # NOTE:
    # The conditional below is hard-coded to always run.
    # Previously this was likely intended to skip completed samples.
    if [ 1 -eq 1 ]; then

        #######################################################################
        # Parse comma-separated SRA IDs into an array
        #######################################################################
        IFS=',' read -ra sra_list <<< "$sra_id"

        # Create (or reuse) sample directory
        mkdir -p "$outdir"

        # Remove any previous contents to avoid mixing old and new runs
        rm -rf $outdir/*

        # Redirect all output to sample-specific log
        exec > >(tee -a "$logfile") 2>&1

        echo "=== Processing ${sam_id} at $(date) ==="

        #######################################################################
        # STEP 1: Download SRA files using NCBI SRA Toolkit
        #
        # Downloads all SRA runs listed for this sample
        #######################################################################
        echo -e "\n######## STEP 1: Downloading SRA files... ########\n"
        ../sratoolkit.3.3.0-ubuntu64/bin/prefetch ${sra_id} || {
            echo "ERROR: prefetch failed for ${sam_id}"
            return 1
        }

        #######################################################################
        # STEP 2: Convert SRA → FASTQ (paired-end)
        #######################################################################
        echo -e "\n######## STEP 2: Converting SRA to FASTQ... ########\n"
        for sra_id in "${sra_list[@]}"; do
            echo "Processing $sra_id..."

            # Handle full SRA files
            if [ -f "./$sra_id/$sra_id.sra" ]; then
                ../sratoolkit.3.3.0-ubuntu64/bin/fasterq-dump \
                    "./$sra_id/$sra_id.sra" \
                    -O "$outdir" \
                    --split-files \
                    --threads 32 || {
                    echo "ERROR: fasterq-dump failed for ${sra_id}"
                    return 1
                }

            # Handle sralite files (sometimes returned by prefetch)
            elif [ -f "./$sra_id/$sra_id.sralite" ]; then
                ../sratoolkit.3.3.0-ubuntu64/bin/fasterq-dump \
                    "./$sra_id/$sra_id.sralite" \
                    -O "$outdir" \
                    --split-files \
                    --threads 32 || {
                    echo "ERROR: fasterq-dump failed for ${sra_id}"
                    return 1
                }
            fi
        done

        # Remove SRA download directory to save space
        rm -r ./$sra_id

        #######################################################################
        # STEP 3: Merge FASTQ files from multiple runs (if present)
        #######################################################################
        echo -e "\n######## STEP 3: Merging FASTQ files... ########\n"
        cd "$outdir"

        # Merge all R1 reads
        if ls *_1.fastq 1> /dev/null 2>&1; then
            cat *_1.fastq > ${sam_id}_1.fastq
            rm $(ls *_1.fastq | grep -v "^${sam_id}_1.fastq$")
        fi

        # Merge all R2 reads (only if paired-end data exists)
        if ls *_2.fastq 1> /dev/null 2>&1; then
            cat *_2.fastq > ${sam_id}_2.fastq
            rm $(ls *_2.fastq | grep -v "^${sam_id}_2.fastq$")
        fi

        #######################################################################
        # STEP 4: Quality control & trimming using fastp
        #
        # Parameters:
        #   --length_required 50   → discard short reads
        #   --dedup                → remove duplicated reads
        #   --average_qual 20      → minimum average base quality
        #   --thread 8             → parallel processing
        #######################################################################
        echo -e "\n######## STEP 4: Running fastp QC...\n"
        # Check if paired-end or single-end
        if [ -f "${sam_id}_2.fastq" ]; then
            # Paired-end processing
            fastp \
                -i ${sam_id}_1.fastq \
                -I ${sam_id}_2.fastq \
                -o ${sam_id}_trimmed_1.fastq \
                -O ${sam_id}_trimmed_2.fastq \
                -h ${sam_id}_fastp.html \
                -j ${sam_id}_fastp.json \
                --length_required 50 \
                --dedup \
                --thread 8 \
                --average_qual 20 || {
                echo "ERROR: fastp failed for ${sam_id}"
                return 1
            }
            rm ${sam_id}_1.fastq ${sam_id}_2.fastq
        else
            # Single-end processing
            fastp \
                -i ${sam_id}_1.fastq \
                -o ${sam_id}_trimmed_1.fastq \
                -h ${sam_id}_fastp.html \
                -j ${sam_id}_fastp.json \
                --length_required 50 \
                --dedup \
                --thread 8 \
                --average_qual 20 || {
                echo "ERROR: fastp failed for ${sam_id}"
                return 1
            }
            rm ${sam_id}_1.fastq
        fi

        echo -e "\n######## STEP 4: fastp completed successfully ########\n"

        #######################################################################
        # STEP 5: Align reads to MTB reference genome using BWA-MEM
        #
        # -M flag: mark shorter split hits as secondary (Picard-compatible)
        # -R: read group information
        #######################################################################
        echo -e "\n######## STEP 5: Running BWA alignment... ########\n"
        # Align reads (handle both single-end and paired-end)
        if [ -f "${sam_id}_trimmed_2.fastq" ]; then
            # Paired-end alignment
            bwa mem \
                -M \
                -R "@RG\tID:${sam_id}\tSM:${sam_id}" \
                -t 4 \
                ../../reference/NC_000962.3.fasta \
                ${sam_id}_trimmed_1.fastq \
                ${sam_id}_trimmed_2.fastq \
                > ${sam_id}.sam || {
                echo "ERROR: BWA alignment failed for ${sam_id}"
                return 1
            }
        else
            # Single-end alignment
            bwa mem \
                -M \
                -R "@RG\tID:${sam_id}\tSM:${sam_id}" \
                -t 4 \
                ../../reference/NC_000962.3.fasta \
                ${sam_id}_trimmed_1.fastq \
                > ${sam_id}.sam || {
                echo "ERROR: BWA alignment failed for ${sam_id}"
                return 1
            }
        fi

        # Remove trimmed FASTQ files to save space
        rm -f *_trimmed_*.fastq

        #######################################################################
        # STEP 6: Convert SAM → sorted BAM
        #######################################################################
        echo -e "\n######## STEP 6: Converting to BAM and sorting... ########\n"
        samtools view -b ${sam_id}.sam \
            | samtools sort -@ 4 \
            > ${sam_id}.bam || {
            echo "ERROR: SAM to BAM conversion failed for ${sam_id}"
            return 1
        }

        # Index BAM file
        samtools index ${sam_id}.bam

        # Generate alignment QC statistics
        samtools flagstat ${sam_id}.bam > ${sam_id}.flagstat.txt

        # Remove SAM file
        rm ${sam_id}.sam

        #######################################################################
        # STEP 7: Remove PCR duplicates using Picard
        #######################################################################
        echo -e "\n######## STEP 7: Removing duplicates with Picard... ########\n"
        picard -Xmx30g MarkDuplicates \
            I=${sam_id}.bam \
            O=${sam_id}_sorted.bam \
            REMOVE_DUPLICATES=true \
            M=${sam_id}.metrics \
            ASSUME_SORT_ORDER=coordinate \
            READ_NAME_REGEX='(?:.*.)?([0-9]+)[^.]*.([0-9]+)[^.]*.([0-9]+)[^.]*$' || {
            echo "ERROR: Picard MarkDuplicates failed for ${sam_id}"
            return 1
        }

        # Index deduplicated BAM
        samtools index ${sam_id}_sorted.bam

        #######################################################################
        # STEP 8: Variant calling using FreeBayes
        #
        # -p 1: haploid genome (bacterial)
        #######################################################################
        echo -e "\n######## STEP 8: Running FreeBayes variant calling... ########\n"

        freebayes-parallel \
            <(fasta_generate_regions.py ../../reference/NC_000962.3.fasta.fai 100000) \
            4 \
            -f ../../reference/NC_000962.3.fasta \
            ${sam_id}_sorted.bam \
            -p 1 \
            > ${sam_id}.vcf || {
            echo "ERROR: FreeBayes failed for ${sam_id}"
            return 1
        }

        # Remove deduplicated BAM to save space
        rm ${sam_id}_sorted.bam

        # Compress VCF
        bgzip ${sam_id}.vcf

        #######################################################################
        # STEP 9: Compute depth and coverage statistics
        #######################################################################
        echo -e "\n######## STEP 9: Computing coverage statistics... ########\n"
        samtools depth -a ${sam_id}.bam > depth.txt

        # Percentage of genome covered at ≥10× depth
        awk '{if ($3 >= 10) covered++} END {print covered/NR*100}' \
            depth.txt >> coverage.txt

        #######################################################################
        # STEP 10: Cleanup intermediate files
        #######################################################################
        echo -e "\n######## STEP 10: Cleaning up intermediate files... ########\n"
        rm -f ${sam_id}.bam
        rm -f ${sam_id}.bam.bai
        rm -f ${sam_id}_sorted.bam.bai
        rm -f ${sam_id}.metrics

        echo -e "\n=== ${sam_id} completed successfully at $(date) ===\n"

        # Return to parent directory
        cd ../

		echo -e "\n##################################################################\n"
    fi
}

###############################################################################
# Export function for GNU Parallel
###############################################################################
echo "#### Processing ###"
export -f process_sample_new

###############################################################################
# Run pipeline in parallel
#
# Input file:
#   ../WHO/resolved_1720.tsv
#
# Columns:
#   {1} → Sample ID
#   {2} → Comma-separated SRA IDs
#
# Parallelisation:
#   -P 15 → process up to 15 samples simultaneously
###############################################################################
cat ../WHO/resolved_1720.tsv \
    | parallel --colsep '\t' -P 1 --joblog parallel_jobs.log \
    process_sample_new {1} {2}

# Alternative inputs (commented out)
# cat who_final_single.tsv | parallel --colsep '\t' -P 15 process_sample_new {1} {2}
