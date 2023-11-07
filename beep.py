import winsound
import time

def sound_alert(frequency, duration):
    winsound.Beep(frequency, duration)

# Example usage: Beep twice with a 1000 Hz frequency and 500 ms duration
for _ in range(5):
    sound_alert(1000, 500)  # Beep at 1000 Hz for 500 milliseconds
    time.sleep(1)  # Wait for 1 second between beeps
