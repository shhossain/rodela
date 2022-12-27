import os
import subprocess
import platform

python_cmd = 'python3' if platform.system() == 'Linux' else 'python'


current_dir = os.path.dirname(os.path.abspath(__file__))
main_file = os.path.join(current_dir, 'main.py')

while True:
    subprocess.call([python_cmd, main_file])
    print('Restarting...')