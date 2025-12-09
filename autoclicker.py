import pyautogui
import time
import sys

print("Starting autoclicker... Press Ctrl+C to stop")
try:
    while True:
        pyautogui.click()
        print(f"Clicked at {time.strftime('%H:%M:%S')}")
        time.sleep(30)
except KeyboardInterrupt:
    print("\nAutoclicker stopped")
