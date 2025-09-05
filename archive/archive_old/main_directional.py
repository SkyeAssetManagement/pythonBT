#!/usr/bin/env python3

import numpy as np
import pandas as pd
from validation_directional import DirectionalValidator
import configparser

def main():
    print("Directional Trading Prediction Model")
    print("=" * 50)
    
    config = configparser.ConfigParser(inline_comment_prefixes='#')
    config.read('config_longonly.ini')
    
    # Load analysis parameters
    edge_threshold_good = float(config['analysis']['edge_threshold_good'])
    recent_predictions_count = int(config['analysis']['recent_predictions_count'])
    results_file = config['output']['results_file']
    model_type = config['model']['model_type']
    
    print(f"Configuration:")
    print(f"- Data file: {config['data']['csv_file']}")
    if 'selected_features' in config['data']:
        print(f"- Selected features: {config['data']['selected_features']}")
    else:
        print(f"- Selected feature: {config['data']['selected_feature']}")
    print(f"- Target column: {config['data']['target_column']}")
    print(f"- Model type: {model_type}")
    print(f"- Target threshold: {config['model']['target_threshold']}")
    print(f"- Vote threshold: {config['model']['vote_threshold']}")
    print(f"- Training size: {config['validation']['train_size']}")
    print(f"- Test size: {config['validation']['test_size']}")
    print()
    
    validator = DirectionalValidator()
    
    print(f"Running {model_type} walk-forward validation...")
    results_df = validator.run_validation()
    
    if len(results_df) == 0:
        print("No valid predictions generated. Check data and configuration.")
        return
    
    print(f"\nValidation completed!")
    print(f"Total observations: {len(results_df)}")
    
    metrics = validator.calculate_directional_metrics(results_df)
    
    # Dynamic labels based on model type
    if model_type == 'longonly':
        signal_name = "LONG"
        direction_name = "UP"
    else:  # shortonly
        signal_name = "SHORT"
        direction_name = "DOWN"
    
    print(f"\n{model_type.title()} Performance Metrics:")
    print("=" * 40)
    print(f"Total observations: {metrics['total_observations']:,}")
    print(f"{signal_name} signals: {metrics['total_trades']:,} ({metrics['trading_frequency']:.1%})")
    print(f"NO TRADE signals: {metrics['no_trade_signals']:,} ({metrics['no_trade_rate']:.1%})")
    print()
    
    print(f"{signal_name} Signal Performance:")
    print(f"- Hit rate when {signal_name}: {metrics['hit_rate']:.3f}")
    print(f"- Avg return when {signal_name}: {metrics['avg_return']:.4f}")
    print(f"- Total P&L: {metrics['total_pnl']:.2f}")
    print()
    
    print("NO TRADE Analysis:")
    print(f"- Would have been profitable: {metrics['no_trade_would_be_profitable']:.3f}")
    print(f"- Avg return if traded: {metrics['no_trade_avg_return']:.4f}")
    print()
    
    print("Base Rates (Random):")
    print(f"- Overall profitable rate: {metrics['overall_profitable_rate']:.3f}")
    print(f"- Overall avg return: {metrics['overall_avg_return']:.4f}")
    print()
    
    print("Edge Analysis:")
    print(f"- Hit rate edge: {metrics['edge']:+.3f}")
    print(f"- Return edge: {metrics['return_edge']:+.4f}")
    
    print("\nRisk/Return Metrics:")
    print(f"- Average monthly P&L: {metrics['avg_monthly_pnl']:.2f}")
    print(f"- Monthly P&L volatility: {metrics['std_monthly_pnl']:.2f}")
    print(f"- Sharpe ratio: {metrics['sharpe_ratio']:.3f}")
    print(f"- Positive months: {metrics['positive_months_pct']:.1%}")
    print(f"- Best month: {metrics['best_month']:.2f}")
    print(f"- Worst month: {metrics['worst_month']:.2f}")
    
    if metrics['edge'] > edge_threshold_good:
        print(f"\nModel shows promising edge: {metrics['edge']:.1%} above base rate!")
    elif metrics['edge'] > 0:
        print(f"\nModel shows slight edge: {metrics['edge']:.1%} above base rate.")
    else:
        print(f"\nModel shows no edge: {metrics['edge']:.1%} vs base rate.")
    
    # Save detailed results
    results_df.to_csv(results_file, index=False)
    print(f"\nDetailed results saved to: {results_file}")
    
    # Show probability distribution
    print(f"\nPrediction Probability Distribution:")
    print(f"- Avg probability when {signal_name}: {results_df[results_df['prediction']==1]['probability'].mean():.3f}")
    print(f"- Avg probability when NO TRADE: {results_df[results_df['prediction']==0]['probability'].mean():.3f}")
    
    # Show some sample recent predictions
    print(f"\nSample Recent Predictions:")
    recent = results_df.tail(recent_predictions_count)[['date', 'prediction', 'probability', 'actual_profitable', 'target_value']]
    for _, row in recent.iterrows():
        signal = signal_name if row['prediction'] == 1 else "NO_TRADE"
        result = "PROFITABLE" if row['actual_profitable'] == 1 else "NOT_PROFITABLE"
        
        # Calculate actual P&L for display
        if row['prediction'] == 1:  # Only show P&L for trades
            if model_type == 'longonly':
                actual_pnl = row['target_value']
            else:  # shortonly
                actual_pnl = -row['target_value']
            print(f"{row['date'].strftime('%Y-%m-%d')}: {signal:8} (p={row['probability']:.2f}) -> {result} (P&L: {actual_pnl:+.3f})")
        else:
            print(f"{row['date'].strftime('%Y-%m-%d')}: {signal:8} (p={row['probability']:.2f}) -> {result} (return: {row['target_value']:+.3f})")

if __name__ == "__main__":
    main()