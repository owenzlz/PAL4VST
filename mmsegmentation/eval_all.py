import argparse
import pdb
import os 



def edit_script(input_file, output_file, src_eval_path, dst_eval_path):
    # Read the content of the target script
    with open(input_file, 'r') as file:
        content = file.read()
    
    # Process the content
    modified_content = content.replace("images/test", "images")
    modified_content = modified_content.replace("labels/test", "labels")
    modified_content = modified_content.replace(src_eval_path, dst_eval_path)

    # Write the modified content back to the script
    with open(output_file, 'w') as file:
        file.write(modified_content)



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="evaluate model on multiple datasets")
    parser.add_argument('--eval_modes', nargs='*', type=str, default=[])
    parser.add_argument("--config_file", default='', type=str)
    parser.add_argument("--checkpoint_file", default='', type=str)
    args = parser.parse_args()

    ori_eval_path = "/sensei-fs/users/lingzzha/projects/AdobePAL/data/metacaf/adobepal_datasets/data2"

    if 'cmgan' in args.eval_modes:
        cmgan_eval_path = "/sensei-fs/users/lingzzha/projects/AdobePAL/data/metacaf/adobepal_datasets/evaluation/cmgan"
        Ecmgan_config_file = args.config_file.replace('.py', '_Ecmgan.py')
        os.remove(Ecmgan_config_file) if os.path.exists(Ecmgan_config_file) else None 
        edit_script(args.config_file, Ecmgan_config_file, ori_eval_path, cmgan_eval_path)
        eval_cmgan_cmd = 'python tools/test.py \
                    %s \
                    %s ' % (Ecmgan_config_file, args.checkpoint_file)
        print(eval_cmgan_cmd)
        pdb.set_trace()
        os.system(eval_cmgan_cmd)
        print("==============================================================")
        print(Ecmgan_config_file.split('/')[-1], 'cmgan')
        print("==============================================================")
    
    if 'supercaf' in args.eval_modes:

        supercaf_eval_path = "/sensei-fs/users/lingzzha/projects/AdobePAL/data/metacaf/adobepal_datasets/evaluation/supercaf"
        Esupercaf_config_file = args.config_file.replace('.py', '_Esupercaf.py')
        os.remove(Esupercaf_config_file) if os.path.exists(Esupercaf_config_file) else None 
        edit_script(args.config_file, Esupercaf_config_file, ori_eval_path, supercaf_eval_path)
        eval_supercaf_cmd = 'python tools/test.py \
                    %s \
                    %s ' % (Esupercaf_config_file, args.checkpoint_file)
        print(eval_supercaf_cmd)
        os.system(eval_supercaf_cmd)
        print("==============================================================")
        print(Esupercaf_config_file.split('/')[-1], 'supercaf')
        print("==============================================================")

    if 'pd2k' in args.eval_modes:

        pd2k_eval_path = "/sensei-fs/users/lingzzha/projects/AdobePAL/data/metacaf/adobepal_datasets/evaluation/pd2k"
        Epd2k_config_file = args.config_file.replace('.py', '_Epd2k.py')
        os.remove(Epd2k_config_file) if os.path.exists(Epd2k_config_file) else None 
        edit_script(args.config_file, Epd2k_config_file, ori_eval_path, pd2k_eval_path)
        eval_pd2k_cmd = 'python tools/test.py \
                    %s \
                    %s ' % (Epd2k_config_file, args.checkpoint_file)
        print(eval_pd2k_cmd)
        os.system(eval_pd2k_cmd)
        print("==============================================================")
        print(Epd2k_config_file.split('/')[-1], 'pd2k')
        print("==============================================================")

    if 'unified1' in args.eval_modes:

        unified1_eval_path = "/sensei-fs/users/lingzzha/projects/AdobePAL/data/metacaf/adobepal_datasets/evaluation/unified1"
        Eunified1_config_file = args.config_file.replace('.py', '_Eunified1.py')
        os.remove(Eunified1_config_file) if os.path.exists(Eunified1_config_file) else None 
        edit_script(args.config_file, Eunified1_config_file, ori_eval_path, unified1_eval_path)
        eval_unified1_cmd = 'python tools/test.py \
                    %s \
                    %s ' % (Eunified1_config_file, args.checkpoint_file)
        print(eval_unified1_cmd)
        os.system(eval_unified1_cmd)
        print("==============================================================")
        print(Eunified1_config_file.split('/')[-1], 'unified1')
        print("==============================================================")

    if 'unified2' in args.eval_modes:

        unified2_eval_path = "/sensei-fs/users/lingzzha/projects/AdobePAL/data/metacaf/adobepal_datasets/evaluation/unified2"
        Eunified2_config_file = args.config_file.replace('.py', '_Eunified2.py')
        os.remove(Eunified2_config_file) if os.path.exists(Eunified2_config_file) else None 
        edit_script(args.config_file, Eunified2_config_file, ori_eval_path, unified2_eval_path)
        eval_unified2_cmd = 'python tools/test.py \
                    %s \
                    %s ' % (Eunified2_config_file, args.checkpoint_file)
        print(eval_unified2_cmd)
        os.system(eval_unified2_cmd)
        print("==============================================================")
        print(Eunified2_config_file.split('/')[-1], 'unified2')
        print("==============================================================")



    