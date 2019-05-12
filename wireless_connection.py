import subprocess
import wifi
from wifi import Cell, Scheme, scan

#results = subprocess.check_output(["netsh", "wlan", "show", "network"])

#print(results.decode('ascii'))

ssids = [cell.ssid for cell in Cell.all('wlan0')]

schemes = list(Scheme.all())

print(schemes)