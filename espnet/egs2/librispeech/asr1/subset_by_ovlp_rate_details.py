 #!/usr/bin/env python

"""
Author: Jiawen Kang (jwkang at se.cuhk.edu.hk)

This script is used to subset the ref and hyp trn files by the ovlp rate.
It requires a utt2rate file to specify the ovlp rate of each utt.
"""

import os
import click


@click.command()
@click.option("--utt2rate", type=click.Path(exists=True), required=True, help="Utt2ovlp file")
@click.option("--ref", type=click.Path(exists=True), required=True, help="Reference trn")
@click.option("--hyp", type=click.Path(exists=True), required=True, help="Hypothsis trn")
@click.option("--out_dir", type=click.Path(), required=True, help="Output directory")
def main(
    utt2rate: click.Path,
    ref: click.Path,
    hyp: click.Path,
    out_dir: click.Path
):

    # 1. group utts by ovlp rate
    part1_name_list = []
    part2_name_list = []
    part3_name_list = []
    part4_name_list = []
    part5_name_list = []
    part6_name_list = []
    part7_name_list = []
    part8_name_list = []
    part9_name_list = []
    part10_name_list = []
    with open(utt2rate, 'r') as f:
        for line in f.readlines():
            name, rate = line.strip().split()
            name = name.split('-', 1)[-1].replace('_', '-')
            if float(rate) <= 0.10:
                part1_name_list.append(name)
            elif float(rate) <= 0.20:
                part2_name_list.append(name)
            elif float(rate) <= 0.3:
                part3_name_list.append(name)
            elif float(rate) <= 0.4:
                part4_name_list.append(name)
            elif float(rate) <= 0.5:
                part5_name_list.append(name)
            elif float(rate) <= 0.6:
                part6_name_list.append(name)
            elif float(rate) <= 0.7:
                part7_name_list.append(name)
            elif float(rate) <= 0.8:
                part8_name_list.append(name)
            elif float(rate) <= 0.9:
                part9_name_list.append(name)
            elif float(rate) <= 1.0:
                part10_name_list.append(name)
            else:
                raise KeyError

    # 2. subset
    f_ref1 = open(os.path.join(out_dir, 'ref_0.10.trn'), 'w')
    f_ref2 = open(os.path.join(out_dir, 'ref_0.20.trn'), 'w')
    f_ref3 = open(os.path.join(out_dir, 'ref_0.30.trn'), 'w')
    f_ref4 = open(os.path.join(out_dir, 'ref_0.40.trn'), 'w')
    f_ref5 = open(os.path.join(out_dir, 'ref_0.50.trn'), 'w')
    f_ref6 = open(os.path.join(out_dir, 'ref_0.60.trn'), 'w')
    f_ref7 = open(os.path.join(out_dir, 'ref_0.70.trn'), 'w')
    f_ref8 = open(os.path.join(out_dir, 'ref_0.80.trn'), 'w')
    f_ref9 = open(os.path.join(out_dir, 'ref_0.90.trn'), 'w')
    f_ref10 = open(os.path.join(out_dir, 'ref_1.00.trn'), 'w')

    # f_hyp1 = open(os.path.join(out_dir, 'hyp_0.20.trn'), 'w')
    # f_hyp2 = open(os.path.join(out_dir, 'hyp_0.50.trn'), 'w')
    # f_hyp3 = open(os.path.join(out_dir, 'hyp_1.00.trn'), 'w')
    f_hyp1 = open(os.path.join(out_dir, 'hyp_0.10.trn'), 'w')
    f_hyp2 = open(os.path.join(out_dir, 'hyp_0.20.trn'), 'w')
    f_hyp3 = open(os.path.join(out_dir, 'hyp_0.30.trn'), 'w')
    f_hyp4 = open(os.path.join(out_dir, 'hyp_0.40.trn'), 'w')
    f_hyp5 = open(os.path.join(out_dir, 'hyp_0.50.trn'), 'w')
    f_hyp6 = open(os.path.join(out_dir, 'hyp_0.60.trn'), 'w')
    f_hyp7 = open(os.path.join(out_dir, 'hyp_0.70.trn'), 'w')
    f_hyp8 = open(os.path.join(out_dir, 'hyp_0.80.trn'), 'w')
    f_hyp9 = open(os.path.join(out_dir, 'hyp_0.90.trn'), 'w')
    f_hyp10 = open(os.path.join(out_dir, 'hyp_1.00.trn'), 'w')
    # test_clean_3mix_0000
    # test-clean-2mix-0002-test-clean-2mix-0002
    # 3. write 
    with open(ref, 'r') as ref:
        for line in ref.readlines():
            name = line.split('(')[1].split(')')[0]
            name = name.split('-',4)[-1]
            if name in part1_name_list:
                f_ref1.write(line)
            elif name in part2_name_list:
                f_ref2.write(line)
            elif name in part3_name_list:
                f_ref3.write(line)
            elif name in part4_name_list:
                f_ref4.write(line)
            elif name in part5_name_list:
                f_ref5.write(line)
            elif name in part6_name_list:
                f_ref6.write(line)
            elif name in part7_name_list:
                f_ref7.write(line)
            elif name in part8_name_list:
                f_ref8.write(line)
            elif name in part9_name_list:
                f_ref9.write(line)
            elif name in part10_name_list:
                f_ref10.write(line)
            else:
                raise KeyError(part1_name_list, name)

    with open(hyp, 'r') as hyp:
        for line in hyp.readlines():
            name = line.split('(')[1].split(')')[0]
            name = name.split('-',4)[-1]
            if name in part1_name_list:
                f_hyp1.write(line)
            elif name in part2_name_list:
                f_hyp2.write(line)
            elif name in part3_name_list:
                f_hyp3.write(line)
            elif name in part4_name_list:
                f_hyp4.write(line)
            elif name in part5_name_list:
                f_hyp5.write(line)
            elif name in part6_name_list:
                f_hyp6.write(line)
            elif name in part7_name_list:
                f_hyp7.write(line)
            elif name in part8_name_list:
                f_hyp8.write(line)
            elif name in part9_name_list:
                f_hyp9.write(line)
            elif name in part10_name_list:
                f_hyp10.write(line)
            else:
                raise KeyError(name)


if __name__ == '__main__':
    main()
