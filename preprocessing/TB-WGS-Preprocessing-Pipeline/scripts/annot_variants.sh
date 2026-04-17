#!/bin/bash

###############################################################################
# Variant annotation pipeline
#
# Steps:
#   1. Filter variants using bcftools (depth & allele frequency)
#   2. Compress and index intermediate VCFs
#   3. Rename reference contig for SnpEff compatibility
#   4. Annotate variants using SnpEff (MTB H37Rv database)
#
# Input per sample:
#   ./<SAMPLE_ID>/<SAMPLE_ID>.vcf.gz
#   ./<SAMPLE_ID>/depth.txt
#
# Output per sample:
#   annot_variants_withfiltering/<SAMPLE_ID>_annot.vcf.gz
###############################################################################

annot_variants()
{
    # Print sample ID (useful for logging when running in parallel)
    echo "${1}"

    # Define sample-specific output directory
    outdir="./${1}"

    # Proceed only if depth file exists and is non-empty
    # This ensures that variant calling was successful earlier
    if [ -s "$outdir/depth.txt" ]; then

        #######################################################################
        # STEP 1: Filter variants
        #
        # Filters applied:
        #   - INFO/DP >= 10  → minimum read depth of 10
        #   - INFO/AF >= 0.9 → high-confidence (near-fixed) variants
        #
        # Output:
        #   annot_variants_withfiltering/<sample>_temp1.vcf
        #######################################################################
        bcftools view \
            -i 'INFO/DP>=10 && INFO/AF>=0.9' \
            "$outdir/${1}.vcf.gz" \
            > "annot_variants_withfiltering/${1}_temp1.vcf"

        # Compress filtered VCF
        bgzip "annot_variants_withfiltering/${1}_temp1.vcf"

        #######################################################################
        # STEP 2: Rename contig ID
        #
        # Original contig name:
        #   NC_000962.3
        #
        # Renamed to:
        #   Chromosome
        #
        # Reason:
        #   SnpEff MTB database expects contig name "Chromosome"
        #######################################################################
        zcat "annot_variants_withfiltering/${1}_temp1.vcf.gz" \
            | sed 's/^NC_000962.3/Chromosome/' \
            > "annot_variants_withfiltering/${1}_temp2.vcf"

        # Compress renamed VCF
        bgzip "annot_variants_withfiltering/${1}_temp2.vcf"

        # Index the VCF for SnpEff compatibility
        bcftools index "annot_variants_withfiltering/${1}_temp2.vcf.gz"

        #######################################################################
        # STEP 3: Variant annotation using SnpEff
        #
        # Database:
        #   Mycobacterium_tuberculosis_h37rv
        #
        # Output:
        #   annot_variants_withfiltering/<sample>_annot.vcf
        #######################################################################
        snpEff Mycobacterium_tuberculosis_h37rv \
            "annot_variants_withfiltering/${1}_temp2.vcf.gz" \
            > "annot_variants_withfiltering/${1}_annot.vcf"

        # Compress annotated VCF
        bgzip "annot_variants_withfiltering/${1}_annot.vcf"

        #######################################################################
        # STEP 4: Cleanup intermediate files
        #######################################################################
        rm \
          "annot_variants_withfiltering/${1}_temp1.vcf.gz" \
          "annot_variants_withfiltering/${1}_temp2.vcf.gz"

        #######################################################################
        # OPTIONAL DOWNSTREAM ANALYSES (commented out)
        #
        # These blocks can be enabled to:
        #   - Count mutations per gene
        #   - Count variant types
        #   - Exclude PE/PPE gene families
        #######################################################################

        # Gene-wise mutation counts
        # zcat annot_variants_withfiltering/${1}_annot.vcf.gz \
        # | grep ANN \
        # | awk -F'ANN=' '{if (NF>1) print $2}' \
        # | cut -d "|" -f 4 \
        # | sort | uniq -c | sort -nr \
        # > annot_variants_withfiltering/${1}_gene_mut_count.txt

        # Variant-type counts
        # zcat annot_variants_withfiltering/${1}_annot.vcf.gz \
        # | grep ANN \
        # | awk -F'ANN=' '{if (NF>1) print $2}' \
        # | cut -d "|" -f 2 \
        # | sort | uniq -c | sort -nr \
        # > annot_variants_withfiltering/${1}_variant_count.txt

        # Gene-wise mutation counts excluding PE/PPE genes
        # zcat annot_variants_withfiltering/${1}_annot.vcf.gz \
        # | grep -v PE_ | grep -v PE[0-9] \
        # | grep ANN \
        # | awk -F'ANN=' '{if (NF>1) print $2}' \
        # | cut -d "|" -f 4 \
        # | sort | uniq -c | sort -nr \
        # > annot_variants_withfiltering/${1}_gene_mut_count_noPE.txt

        # Variant-type counts excluding PE/PPE genes
        # zcat annot_variants_withfiltering/${1}_annot.vcf.gz \
        # | grep -v PE_ | grep -v PE[0-9] \
        # | grep ANN \
        # | awk -F'ANN=' '{if (NF>1) print $2}' \
        # | cut -d "|" -f 2 \
        # | sort | uniq -c | sort -nr \
        # > annot_variants_withfiltering/${1}_variant_count_noPE.txt

    fi
}

###############################################################################
# Export function so GNU Parallel can access it
###############################################################################
export -f annot_variants

###############################################################################
# Run annotation in parallel
#
# Input file:
#   WHO_49266_final_ids.csv
#
# Assumes:
#   - Sample ID is in column 1
#   - 20 samples processed in parallel
###############################################################################
cat WHO_49266_final_ids.csv \
    | cut -d "," -f 1 \
    | parallel -P 20 annot_variants
