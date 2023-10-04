import argparse
import pdb
import os 



def edit_script(input_file, output_file, src_data, dst_data):
    # Read the content of the target script
    with open(input_file, 'r') as file:
        content = file.read()
    
    # Process the content
    modified_content = content.replace(src_data, dst_data)

    # Write the modified content back to the script
    with open(output_file, 'w') as file:
        file.write(modified_content)



if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--src_config_file", default='', type=str)
    parser.add_argument("--src_data", default='', type=str)
    parser.add_argument("--dst_data", default='', type=str)
    args = parser.parse_args()
    
    dst_config_file = args.src_config_file.replace(args.src_data, args.dst_data)
    edit_script(args.src_config_file, dst_config_file, args.src_data, args.dst_data)

