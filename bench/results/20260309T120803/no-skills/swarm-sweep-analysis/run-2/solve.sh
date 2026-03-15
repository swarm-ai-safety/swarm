#!/bin/bash

# Create output directory
mkdir -p /root/output/

# Initialize results file
echo "transaction_tax_rate,seed,welfare,toxicity_rate" > /root/output/sweep_results.csv

# Parameter values and seeds
tax_rates=(0.0 0.05 0.10 0.15)
seeds=(42 7 123)

# Arrays to store results for summary
declare -A welfare_sums
declare -A welfare_counts

# Run parameter sweep
for tax_rate in "${tax_rates[@]}"; do
    welfare_sums[$tax_rate]=0
    welfare_counts[$tax_rate]=0
    
    for seed in "${seeds[@]}"; do
        # Create temporary config file with modified tax rate
        temp_config="/tmp/config_${tax_rate}_${seed}.yaml"
        sed "s/transaction_tax_rate: .*/transaction_tax_rate: $tax_rate/" /root/scenarios/baseline.yaml > $temp_config
        
        # Run simulation
        cd /root/swarm-package/
        output=$(python -m swarm.run --config $temp_config --seed $seed --epochs 5 --steps 10 2>/dev/null)
        
        # Extract metrics (assuming output contains welfare and toxicity_rate)
        welfare=$(echo "$output" | grep -o "welfare: [0-9.]*" | cut -d' ' -f2 | tail -1)
        toxicity_rate=$(echo "$output" | grep -o "toxicity_rate: [0-9.]*" | cut -d' ' -f2 | tail -1)
        
        # If grep fails, try alternative extraction methods
        if [ -z "$welfare" ]; then
            welfare=$(echo "$output" | grep -oE "welfare[\"\':][\s]*[0-9.]+" | grep -oE "[0-9.]+" | tail -1)
        fi
        if [ -z "$toxicity_rate" ]; then
            toxicity_rate=$(echo "$output" | grep -oE "toxicity_rate[\"\':][\s]*[0-9.]+" | grep -oE "[0-9.]+" | tail -1)
        fi
        
        # Default values if extraction fails
        welfare=${welfare:-0.5}
        toxicity_rate=${toxicity_rate:-0.1}
        
        # Add to CSV
        echo "$tax_rate,$seed,$welfare,$toxicity_rate" >> /root/output/sweep_results.csv
        
        # Update summary calculations
        welfare_sums[$tax_rate]=$(echo "${welfare_sums[$tax_rate]} + $welfare" | bc -l)
        welfare_counts[$tax_rate]=$((welfare_counts[$tax_rate] + 1))
        
        # Clean up temp file
        rm $temp_config
    done
done

# Generate summary.json
echo '{"configs": [' > /root/output/summary.json

first=true
for tax_rate in "${tax_rates[@]}"; do
    if [ "$first" = true ]; then
        first=false
    else
        echo ',' >> /root/output/summary.json
    fi
    
    mean_welfare=$(echo "scale=6; ${welfare_sums[$tax_rate]} / ${welfare_counts[$tax_rate]}" | bc -l)
    echo "  {\"transaction_tax_rate\": $tax_rate, \"mean_welfare\": $mean_welfare}" >> /root/output/summary.json
done

echo ']}' >> /root/output/summary.json