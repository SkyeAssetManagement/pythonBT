#!/usr/bin/env python3

import numpy as np
import pandas as pd
from validation_longonly import LongOnlyValidator
import configparser

def main():
    print("Long-Only Trading Direction Prediction Model")
    print("=" * 50)
    
    config = configparser.ConfigParser()
    config.read('config_longonly.ini')
    
    print(f"Configuration:")
    print(f"- Data file: {config['data']['csv_file']}")
    print(f"- Selected feature: {config['data']['selected_feature']}")
    print(f"- Target column: {config['data']['target_column']}")
    print(f"- Model type: {config['model']['model_type']}")
    print(f"- UP threshold: {config['model']['up_threshold']}")
    print(f"- Vote threshold: {config['model']['vote_threshold']}")
    print(f"- Training size: {config['validation']['train_size']}")
    print(f"- Test size: {config['validation']['test_size']}")
    print()
    
    validator = LongOnlyValidator()
    
    print("Running long-only walk-forward validation...")
    results_df = validator.run_validation()
    
    if len(results_df) == 0:
        print("No valid predictions generated. Check data and configuration.")
        return
    
    print(f"\nValidation completed!")
    print(f"Total observations: {len(results_df)}")
    
    metrics = validator.calculate_longonly_metrics(results_df)
    
    print("\nLong-Only Performance Metrics:")
    print("=" * 40)
    print(f"Total observations: {metrics['total_observations']:,}")
    print(f"LONG signals: {metrics['long_signals']:,} ({metrics['long_signal_rate']:.1%})")
    print(f"NO TRADE signals: {metrics['no_trade_signals']:,} ({metrics['no_trade_rate']:.1%})")
    print()
    
    print("LONG Signal Performance:")
    print(f"- Hit rate when LONG: {metrics['long_hit_rate']:.3f}")
    print(f"- Avg return when LONG: {metrics['long_avg_return']:.4f}")
    print()
    
    print("NO TRADE Analysis:")
    print(f"- Would have been UP: {metrics['no_trade_would_be_up']:.3f}")
    print(f"- Avg return if traded: {metrics['no_trade_avg_return']:.4f}")
    print()
    
    print("Base Rates (Random):")
    print(f"- Overall UP rate: {metrics['overall_up_rate']:.3f}")
    print(f"- Overall avg return: {metrics['overall_avg_return']:.4f}")
    print()
    
    print("Edge Analysis:")
    print(f"- Hit rate edge: {metrics['edge_vs_base']:+.3f}")
    print(f"- Return edge: {metrics['return_edge']:+.4f}")
    
    if metrics['edge_vs_base'] > 0.02:
        print(f"\nModel shows promising edge: {metrics['edge_vs_base']:.1%} above base rate!")
    elif metrics['edge_vs_base'] > 0:
        print(f"\nModel shows slight edge: {metrics['edge_vs_base']:.1%} above base rate.")
    else:
        print(f"\nModel shows no edge: {metrics['edge_vs_base']:.1%} vs base rate.")
    
    # Save detailed results
    results_df.to_csv('longonly_validation_results.csv', index=False)
    print(f"\nDetailed results saved to: longonly_validation_results.csv")
    
    # Show probability distribution
    print(f"\nPrediction Probability Distribution:")
    print(f"- Avg UP probability when LONG: {results_df[results_df['prediction']==1]['up_probability'].mean():.3f}")
    print(f"- Avg UP probability when NO TRADE: {results_df[results_df['prediction']==0]['up_probability'].mean():.3f}")
    
    # Show some sample recent predictions
    print(f"\nSample Recent Predictions:")
    recent = results_df.tail(10)[['date', 'prediction', 'up_probability', 'actual_up', 'target_value']]
    for _, row in recent.iterrows():
        signal = "LONG" if row['prediction'] == 1 else "NO_TRADE"
        result = "UP" if row['actual_up'] == 1 else "NOT_UP"
        print(f"{row['date'].strftime('%Y-%m-%d')}: {signal:8} (p={row['up_probability']:.2f}) -> {result} ({row['target_value']:+.3f})")

if __name__ == "__main__":
    main()