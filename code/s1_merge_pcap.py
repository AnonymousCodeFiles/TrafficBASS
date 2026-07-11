import os
import shutil
import subprocess
import argparse


def merge_pcap_files(source_dir, target_dir, mergecap_cmd='mergecap'):
    os.makedirs(target_dir, exist_ok=True)

    for sub_dir in os.listdir(source_dir):
        sub_dir_path = os.path.join(source_dir, sub_dir)

        if os.path.isdir(sub_dir_path):
            output_pcap_file = os.path.join(target_dir, f"{sub_dir}.pcap")

            if os.path.exists(output_pcap_file):
                print(f"Skipping: {output_pcap_file} already exists")
                continue

            pcap_files = [os.path.join(sub_dir_path, f) for f in os.listdir(sub_dir_path) if f.endswith('.pcap')]

            if pcap_files:
                valid_pcap_files = []

                for pcap_file in pcap_files:
                    try:
                        subprocess.run([mergecap_cmd, '-w', os.devnull, pcap_file], check=True)
                        valid_pcap_files.append(pcap_file)
                    except subprocess.CalledProcessError as e:
                        print(f"Skipping invalid file: {pcap_file}, error: {e}")

                if valid_pcap_files:
                    try:
                        command = [mergecap_cmd, '-w', output_pcap_file] + valid_pcap_files
                        subprocess.run(command, check=True)
                        print(f"Merge complete: {output_pcap_file}")
                    except subprocess.CalledProcessError as e:
                        print(f"Merge failed: {output_pcap_file}, error: {e}")
                else:
                    print(f"No valid pcap files in subdirectory {sub_dir_path}")
            else:
                print(f"No pcap files in subdirectory {sub_dir_path}")
        else:
            print(f"{sub_dir_path} is not a directory")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge sub-directory PCAP files")
    parser.add_argument('--source_dir', required=True, help='Source directory containing sub-directories with PCAP files')
    parser.add_argument('--target_dir', required=True, help='Output directory for merged PCAP files')
    parser.add_argument('--mergecap', default='mergecap',
                        help='Path to mergecap executable (default: search in PATH)')
    args = parser.parse_args()

    merge_pcap_files(args.source_dir, args.target_dir, args.mergecap)
