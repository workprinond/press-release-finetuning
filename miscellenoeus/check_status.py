# monitor_training.py
import os
import glob
import pandas as pd
import time
from datetime import datetime

def monitor_progress():
    """Monitor training progress by reading the CSV file"""
    
    # Find the latest training folder
    folders = glob.glob('bmw_finetune_results_*')
    if not folders:
        print("No training folder found yet...")
        return None
    
    latest_folder = max(folders)  # Gets most recent by timestamp
    csv_file = os.path.join(latest_folder, "training_history.csv")
    
    return csv_file

def show_latest_status():
    """Show the latest training status"""
    
    csv_file = monitor_progress()
    if not csv_file or not os.path.exists(csv_file):
        print("Training hasn't started creating logs yet...")
        return
    
    try:
        # Read the CSV file
        df = pd.read_csv(csv_file)
        
        print(f"\n📊 Training Progress - {datetime.now().strftime('%H:%M:%S')}")
        print("=" * 50)
        print(f"Total steps: {len(df):,}")
        
        if len(df) > 0:
            # Show latest stats
            latest = df.iloc[-1]
            print(f"Latest step: {latest['step']}")
            print(f"Latest loss: {latest['train_loss']:.4f}")
            print(f"Latest learning rate: {latest['learning_rate']:.6f}")
            
            # Show progress if we have enough data
            if len(df) > 10:
                print(f"\n📈 Progress:")
                print(f"  First loss: {df['train_loss'].iloc[0]:.4f}")
                print(f"  Min loss so far: {df['train_loss'].min():.4f}")
                print(f"  Average loss: {df['train_loss'].mean():.4f}")
                
                # Estimate time remaining
                if len(df) > 100:
                    avg_time_per_step = 1.8  # ~1.8 seconds per step from your output
                    total_steps_estimated = 3791 * 3  # 3791 batches × 3 epochs
                    steps_done = len(df)
                    steps_left = total_steps_estimated - steps_done
                    time_left = steps_left * avg_time_per_step / 3600  # hours
                    
                    print(f"\n⏱️  Estimated:")
                    print(f"  Steps completed: {steps_done:,}/{total_steps_estimated:,}")
                    print(f"  Time remaining: ~{time_left:.1f} hours")
        
    except Exception as e:
        print(f"Error reading CSV: {e}")

def continuous_monitor(interval_seconds=30):
    """Continuously monitor training progress"""
    print("🚀 Starting continuous monitoring...")
    print("Press Ctrl+C to stop\n")
    
    try:
        while True:
            os.system('cls' if os.name == 'nt' else 'clear')
            show_latest_status()
            time.sleep(interval_seconds)
    except KeyboardInterrupt:
        print("\n\n⏹️  Monitoring stopped")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "once":
        show_latest_status()
    else:
        continuous_monitor()