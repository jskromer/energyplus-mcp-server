import subprocess
import os

def run_energyplus_simulation(idf_path, epw_path, output_dir):
    if not os.path.exists(idf_path):
        print(f"❌ IDF file not found: {idf_path}")
        return
    if not os.path.exists(epw_path):
        print(f"❌ EPW file not found: {epw_path}")
        return
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    command = ["energyplus", "-r", "-w", epw_path, "-d", output_dir, idf_path]

    try:
        print(f"🚀 Running simulation: {idf_path}")
        subprocess.run(command, check=True)
        print(f"✅ Simulation complete. Output saved to: {output_dir}")
    except subprocess.CalledProcessError as e:
        print(f"❌ Simulation failed: {e}")

if __name__ == "__main__":
    run_energyplus_simulation("../idf_files/test.idf", "../weather_files/USA.epw", "../outputs/sim_1")
