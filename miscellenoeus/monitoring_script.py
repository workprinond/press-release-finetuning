# monitor.py - Save this as a .py file
import json
import os
import time
import glob
from datetime import datetime

def monitor_training():
    """Monitor training progress from Python"""
    print("🚀 Training Progress Monitor")
    print("=" * 50)
    
    # Find latest output folder
    folders = glob.glob('bmw_finetune_results_*')
    if not folders:
        print("❌ No training folder found!")
        return
    
    latest_folder = max(folders)  # Gets the newest by timestamp
    print(f"📁 Monitoring folder: {latest_folder}")
    print("Press Ctrl+C to stop monitoring\n")
    
    try:
        while True:
            current_time = datetime.now().strftime('%H:%M:%S')
            
            # Check training history
            history_file = os.path.join(latest_folder, "training_history.csv")
            if os.path.exists(history_file):
                with open(history_file, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                
                if len(lines) > 1:
                    # Get latest step and loss
                    latest_line = lines[-1].strip()
                    if ',' in latest_line:
                        parts = latest_line.split(',')
                        if len(parts) >= 2:
                            step = parts[0]
                            loss = parts[1]
                            lr = parts[2] if len(parts) > 2 else "N/A"
                            print(f"[{current_time}] Step: {step}, Loss: {loss}, LR: {lr}")
            
            # Also check for eval loss
            eval_file = os.path.join(latest_folder, "eval_history.csv")
            if os.path.exists(eval_file):
                with open(eval_file, 'r', encoding='utf-8') as f:
                    eval_lines = f.readlines()
                if len(eval_lines) > 1:
                    print(f"📊 Eval history: {len(eval_lines)-1} epochs completed")
            
            time.sleep(30)  # Check every 30 seconds
            
    except KeyboardInterrupt:
        print("\n\n⏹️  Monitoring stopped")
    
    except FileNotFoundError:
        print(f"⚠️  File not found yet. Training might be starting...")
        time.sleep(10)

def show_summary():
    """Show summary of training progress"""
    folders = glob.glob('bmw_finetune_results_*')
    if not folders:
        print("No training results found!")
        return
    
    latest = max(folders)
    print(f"\n📊 Summary for: {latest}")
    print("=" * 50)
    
    # Check files
    files = {
        "Training History": "training_history.csv",
        "Eval History": "eval_history.csv", 
        "Config": "training_config.json",
        "Test Metrics": "test_metrics.json",
        "Samples": "sample_generations.json"
    }
    
    for name, filename in files.items():
        path = os.path.join(latest, filename)
        if os.path.exists(path):
            size = os.path.getsize(path) / 1024  # KB
            print(f"✅ {name}: {filename} ({size:.1f} KB)")
        else:
            print(f"⏳ {name}: {filename} (not created yet)")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "summary":
        show_summary()
    else:
        monitor_training()