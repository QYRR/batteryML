import subprocess
import os
import sys

# Get the absolute path of the current script directory
current_script_dir = os.path.dirname(os.path.abspath(__file__))

# ================= Path Configuration =================
# Virtual environment path (adjust according to actual location)
venv_rel_path = "../venvSANDIA"  # Relative path from the current script directory
venv_abs_path = os.path.normpath(os.path.join(current_script_dir, venv_rel_path))

# Python interpreter path
venv_python = os.path.join(venv_abs_path, "bin", "python3") if sys.platform != "win32" \
    else os.path.join(venv_abs_path, "Scripts", "python.exe")

# Target script path
target_script = os.path.join(current_script_dir, "lightgbm_various_params.py")

# ================= Path Validation =================
def validate_paths():
    """Validate if all critical paths exist"""
    missing = []
    
    if not os.path.exists(venv_python):
        missing.append(f"Virtual environment Python interpreter: {venv_python}")
    
    if not os.path.exists(target_script):
        missing.append(f"Target script: {target_script}")
    
    if missing:
        msg = "Critical paths missing:\n" + "\n".join(missing)
        raise FileNotFoundError(msg)

# ================= Execution Logic =================
def run_experiments(parameters):
    """Run experiments iterating through parameters"""
    for param in parameters:
        exec_times = 1
        for i in range(exec_times):
            print(f"\n=== Parameter {param}, Run {i+1}/{exec_times} started ===")
            
            # Build the command
            command = [
                venv_python,
                target_script,
                "--sequence_length",  # Use named parameter
                str(param)
            ]
            
            try:
                # Execute the command
                result = subprocess.run(
                    command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    check=True,
                    # timeout=300  # 5-minute timeout
                )
                
                # Handle the output
                # Full output handling
                print(f"\n=== Parameter {param}, Run {i+1}/{exec_times} completed - Full Output ===")
                print("Standard Output Content:")
                print(result.stdout)  # Print all content directly

                # if result.stderr:
                #     print("Standard Error Content:")
                #     print(result.stderr)

                print("="*50 + "\n")
                
            except subprocess.CalledProcessError as e:
                print(f"Execution failed! Return code: {e.returncode}")
                print("Error Output:\n", e.stderr)
                # You can choose break or continue
                continue
                
            except subprocess.TimeoutExpired:
                print("Error: Execution timed out (300 seconds)")
                continue
                
            except Exception as e:
                print(f"Unknown error: {str(e)}")
                continue

# ================= Main Program =================

try:
    # Validate paths
    validate_paths()
    
    # Print configuration information
    print(f"Virtual environment Python path: {venv_python}")
    print(f"Target script path: {target_script}")
    print(f"Current working directory: {os.getcwd()}")
    
    # Run parameter experiments
    parameters = [
        # [5,10,20,30,50],
        # [5,10,15,20,30,40,50,70,100],
        # [3,5,7,10,15,20,30,40,50,70,100]
        0
    ]
    run_experiments(parameters)
    
except Exception as e:
    print(f"Initialization failed: {str(e)}")
    sys.exit(1)
