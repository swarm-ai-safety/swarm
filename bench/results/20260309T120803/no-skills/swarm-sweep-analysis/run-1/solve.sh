#!/bin/bash

# Create output directory
mkdir -p /root/output/

# Initialize CSV file with header
echo "transaction_tax_rate,seed,welfare,toxicity_rate" > /root/output/sweep_results.csv

# Initialize summary data
echo '{"configs":[' > /root/output/summary.json

# Tax rates to sweep
tax_rates=(0.0 0.05 0.10 0.15)
seeds=(42 7 123)

# Temporary files for collecting results
temp_results=$(mktemp)
temp_summary=$(mktemp)

config_count=0

# Run parameter sweep
for tax_rate in "${tax_rates[@]}"; do
    welfare_sum=0
    run_count=0
    
    for seed in "${seeds[@]}"; do
        # Run simulation with modified tax rate
        cd /root/swarm-package/
        
        # Create temporary config file with modified tax rate
        temp_config=$(mktemp)
        sed "s/transaction_tax_rate:.*/transaction_tax_rate: $tax_rate/" /root/scenarios/baseline.yaml > $temp_config
        
        # Run simulation
        result=$(python -m swarm.cli --config $temp_config --seed $seed --epochs 5 --steps 10 --output-format json)
        
        # Extract metrics (assuming JSON output format)
        welfare=$(echo "$result" | python -c "import sys, json; data=json.load(sys.stdin); print(data.get('welfare', 0))")
        toxicity_rate=$(echo "$result" | python -c "import sys, json; data=json.load(sys.stdin); print(data.get('toxicity_rate', 0))")
        
        # Add to CSV
        echo "$tax_rate,$seed,$welfare,$toxicity_rate" >> /root/output/sweep_results.csv
        
        # Accumulate for summary
        welfare_sum=$(echo "$welfare_sum + $welfare" | bc -l)
        run_count=$((run_count + 1))
        
        # Clean up temp config
        rm $temp_config
    done
    
    # Calculate mean welfare for this config
    mean_welfare=$(echo "scale=6; $welfare_sum / $run_count" | bc -l)
    
    # Add to summary JSON
    if [ $config_count -gt 0 ]; then
        echo "," >> $temp_summary
    fi
    echo "{\"transaction_tax_rate\":$tax_rate,\"mean_welfare\":$mean_welfare}" >> $temp_summary
    
    config_count=$((config_count + 1))
done

# Complete summary JSON
cat $temp_summary >> /root/output/summary.json
echo ']}' >> /root/output/summary.json

# Clean up temporary files
rm -f $temp_results $temp_summary

echo "Parameter sweep completed. Results saved to /root/output/"