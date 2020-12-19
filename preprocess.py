import os
import hparams as hp
import argparse

if hp.dataset == "LJSpeech":
    from data import ljspeech as Dataset
elif hp.dataset == "Blizzard2013":
    from data import blizzard2013 as Dataset
elif hp.dataset == "VCTK":
    from data import vctk as Dataset
elif hp.dataset == "LibriTTS":
    from data import libritts as Dataset
else:
    raise NotImplementedError("You should specify the dataset in hparams.py\
                              and write a corresponding file in data/")
                              
class Preprocessor():
    def __init__(self, args):
        self.args = args
        self.in_dir  = hp.data_path
        self.out_dir = hp.preprocessed_path
        self.mfa_path = hp.mfa_path
    
    def exec(self):
        self.print_message()
        key_input = ""
        while key_input  not in ["y", "Y", "n", "N"]:
            key_input = input("Proceed? ([y/n])? ")
            
        if key_input in ['y', 'Y']:
            self.make_output_dirs(force=False)
            if self.args.prepare_align:
                print("Preparing alignment text data...")
                self.prepare_align()
                
            if self.args.mfa:
                print("Performing Montreal Force Alignment...")
                self.mfa()
                
            if self.args.create_dataset:
                print("Creating Training and Validation Dataset...")
                self.create_dataset()
            
    def print_message(self):
        print("\n")
        print("------ Preprocessing ------")
        print(f"* Dataset     : {hp.dataset}")
        print(f"* Data path   : {self.in_dir}")
        print(f"* Output path : {self.out_dir}")
        print("\n")
        print("The following will be executed:")
        
        if self.args.prepare_align:
            print("\t* Preparing Alignment Data")
        if self.args.mfa:
            print("\t* Montreal Force Alignmnet")
        if self.args.create_dataset:
            print("\t* Creating Training Dataset")
        print("\n")
            
    def make_output_dirs(self, force=False):
        out_dir = self.out_dir
        if self.args.mfa:
            mfa_out_dir = os.path.join(out_dir, "TextGrid")
            os.makedirs(mfa_out_dir, exist_ok=force)
        
        mel_out_dir = os.path.join(out_dir, "mel")
        os.makedirs(mel_out_dir, exist_ok=force)
        
        ali_out_dir = os.path.join(out_dir, "alignment")
        os.makedirs(ali_out_dir, exist_ok=force)
        
        f0_out_dir = os.path.join(out_dir, "f0")
        os.makedirs(f0_out_dir, exist_ok=force)
        
        energy_out_dir = os.path.join(out_dir, "energy")
        os.makedirs(energy_out_dir, exist_ok=force)
        
    ### Preprocessing ###
    def create_dataset(self):
        '''
        1. train and val meta will be obtained here
        2. during "build_fron_path", alignment, f0, energy and mel data will be created
        '''
        in_dir = self.in_dir
        out_dir = self.out_dir
        
        train, val = Dataset.build_from_path(in_dir, out_dir)
        with open(os.path.join(out_dir, 'train.txt'), 'w', encoding='utf-8') as f:
            for m in train:
                f.write(m + '\n')
        with open(os.path.join(out_dir, 'val.txt'), 'w', encoding='utf-8') as f:
            for m in val:
                f.write(m + '\n')
                
    ### Prepare Align
    def prepare_align(self):
        in_dir = self.in_dir
        Dataset.prepare_align(in_dir)
    
    ### MFA ###
    def mfa(self):
        in_dir = self.in_dir
        out_dir = self.out_dir
        mfa_path = self.mfa_path
        
        mfa_out_dir  = os.path.join(out_dir, "TextGrid")
        mfa_bin_path = os.path.join(mfa_path, "bin", "mfa_align")
        mfa_pretrain_path = os.path.join(mfa_path, "pretrained_models", "librispeech-lexicon.txt")
        cmd = f"{mfa_bin_path} {in_dir} {mfa_pretrain_path} english {mfa_out_dir} -j 8"
        os.system(cmd)    

def main(args):
    P = Preprocessor(args)
    P.exec()
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare_align', action="store_true", default=False)
    parser.add_argument('--mfa', action="store_true", default=False)
    parser.add_argument('--create_dataset', action="store_true", default=False)
    args = parser.parse_args()
    
    main(args)
