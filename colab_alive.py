# Source - https://stackoverflow.com/a
# Posted by Seyon Seyon, modified by community. See post 'Timeline' for change history
# Retrieved 2025-12-08, License - CC BY-SA 4.0

from pynput.mouse import Button, Controller
import time

mouse = Controller()

while True:
    mouse.click(Button.left, 1)
    time.sleep(30)


# import pyautogui
# import time
# import sys

# print("Starting autoclicker... Press Ctrl+C to stop")

# try:
#     while True:
#         pyautogui.click()
#         print(f"Clicked at {time.strftime('%H:%M:%S')}")
#         time.sleep(30)
# except KeyboardInterrupt:
#     print("\nAutoclicker stopped")