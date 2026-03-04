export QT_XCB_GL_INTEGRATION=${QT_XCB_GL_INTEGRATION:-software}
set -e
set -u
set -o pipefail

. path.sh

if [ $# -ne 1 ]; then
    echo "Error: decode_root path is required."
    echo "Usage: $0 <decode_root>"
    echo "Example: $0 /mnt/sdb/work/exp/decode_result"
    exit 1
fi

decode_root=$1

echo "Using decode_root: ${decode_root}"

setname=test  # dev or test

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

progress() {
	percent=$(($1*100/$2))
	curr=$(($percent*4/10))
	left=$((40-$curr))

	printf "\rProgress : [%${curr}s%${left}s] %d%%" "#" "#" $percent
}

for mode in "2mix" "3mix"; do
    echo "Start $mode"
    decoding_path=$decode_root/${setname}_clean_${mode}
    out_dir=$decode_root/pi_scoring/${setname}_clean_${mode}
       
    log "Output dir: ${out_dir}"
    
    # 1: prepare output path
    mkdir -p $out_dir
    cp $0 $out_dir
    cp $decoding_path/score_wer/ref.trn $out_dir/ref_utt.trn
    cp $decoding_path/score_wer/hyp.trn $out_dir/hyp.trn
    ref_utt=$out_dir/ref_utt.trn
    hyp=$out_dir/hyp.trn
    
    rm -rf $out_dir/tmp
    rm -f $out_dir/{best_ref.trn,best_hyp.trn}
    
    # 2: splitting ref and hyp
    log "Splitting TRN."
    python3 search_best_permutation.py \
    		--ref_utt $ref_utt \
    		--hyp $hyp \
    		--out_dir $out_dir
    
    #- out_dir:
    #	- tmp:
    #	|	- utt0:
    #	|	|	- permu0 (utt-based)
    #	|	|	|	- ref.trn
    #	|	|	|	- hyp.trn
    #	|	|	|	- result.txt
    #	|	|	- permu1 (spkr-based)
    #	|	|	|	...
    #	|	|	- wers
    #	|	- utt1:
    #	|	|	...
    #	- best_ref.trn
    #	- best_hyp.trn
    
    # 3: scoring to get final error rate
    sclite \
    		-r $out_dir/best_ref.trn trn \
    		-h $out_dir/best_hyp.trn trn \
    		-i rm -o all stdout > $out_dir/result_permu.txt

    head -n 20 $out_dir/result_permu.txt

    # 4: Subscoring: subset utts into groups of different overlap rate
    # out_dir:
    #  |- ovlp_subscore:
    #         |- ref_0.20.trn
    #         |- ref_0.50.trn
    #             ...
    log "Subset ${setname}_clean_${mode} into ovlp group {0.20, 0.50, 1.00}."
    mkdir -p $out_dir/ovlp_subscore
    python subset_by_ovlp_rate.py \
            --utt2rate files/librispeechmix/${setname}_clean_$mode/utt2ovlp_rate \
            --ref $out_dir/best_ref.trn \
            --hyp $out_dir/best_hyp.trn \
            --out_dir $out_dir/ovlp_subscore

    # # ovlp <=0.25:
    for rate in "0.20" "0.50" "1.00";do
        sclite \
                -r $out_dir/ovlp_subscore/ref_${rate}.trn \
                -h $out_dir/ovlp_subscore/hyp_${rate}.trn \
                -i rm -o all stdout > $out_dir/result_${rate}.txt
    done

done


for mode in "1mix"; do
    echo "Start $mode"
    decoding_path=$decode_root/${setname}_clean_${mode}
    out_dir=$decode_root/pi_scoring/${setname}_clean_${mode}
       
    log "Output dir: ${out_dir}"
    
    # 1: prepare output path
    mkdir -p $out_dir
    cp $0 $out_dir
    cp $decoding_path/score_wer/ref.trn $out_dir/ref_utt.trn
    cp $decoding_path/score_wer/hyp.trn $out_dir/hyp.trn
    ref_utt=$out_dir/ref_utt.trn
    hyp=$out_dir/hyp.trn
    
    rm -rf $out_dir/tmp
    rm -f $out_dir/{best_ref.trn,best_hyp.trn}
    
    # 2: splitting ref and hyp
    log "Splitting TRN."
    python3 search_best_permutation.py \
    		--ref_utt $ref_utt \
    		--hyp $hyp \
    		--out_dir $out_dir
    
    #- out_dir:
    #	- tmp:
    #	|	- utt0:
    #	|	|	- permu0 (utt-based)
    #	|	|	|	- ref.trn
    #	|	|	|	- hyp.trn
    #	|	|	|	- result.txt
    #	|	|	- permu1 (spkr-based)
    #	|	|	|	...
    #	|	|	- wers
    #	|	- utt1:
    #	|	|	...
    #	- best_ref.trn
    #	- best_hyp.trn
    
    # 3: scoring to get final error rate
    sclite \
    		-r $out_dir/best_ref.trn trn \
    		-h $out_dir/best_hyp.trn trn \
    		-i rm -o all stdout > $out_dir/result_permu.txt

    head -n 20 $out_dir/result_permu.txt
done


setname=dev  # dev or test

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

progress() {
	percent=$(($1*100/$2))
	curr=$(($percent*4/10))
	left=$((40-$curr))

	printf "\rProgress : [%${curr}s%${left}s] %d%%" "#" "#" $percent
}

for mode in "2mix2" "3mix"; do
    echo "Start $mode"
    decoding_path=$decode_root/${setname}_clean_${mode}
    out_dir=$decode_root/pi_scoring/${setname}_clean_${mode}
       
    log "Output dir: ${out_dir}"
    
    # 1: prepare output path
    mkdir -p $out_dir
    cp $0 $out_dir
    cp $decoding_path/score_wer/ref.trn $out_dir/ref_utt.trn
    cp $decoding_path/score_wer/hyp.trn $out_dir/hyp.trn
    ref_utt=$out_dir/ref_utt.trn
    hyp=$out_dir/hyp.trn
    
    rm -rf $out_dir/tmp
    rm -f $out_dir/{best_ref.trn,best_hyp.trn}
    
    # 2: splitting ref and hyp
    log "Splitting TRN."
    python3 search_best_permutation.py \
    		--ref_utt $ref_utt \
    		--hyp $hyp \
    		--out_dir $out_dir
    
    #- out_dir:
    #	- tmp:
    #	|	- utt0:
    #	|	|	- permu0 (utt-based)
    #	|	|	|	- ref.trn
    #	|	|	|	- hyp.trn
    #	|	|	|	- result.txt
    #	|	|	- permu1 (spkr-based)
    #	|	|	|	...
    #	|	|	- wers
    #	|	- utt1:
    #	|	|	...
    #	- best_ref.trn
    #	- best_hyp.trn
    
    # 3: scoring to get final error rate
    sclite \
    		-r $out_dir/best_ref.trn trn \
    		-h $out_dir/best_hyp.trn trn \
    		-i rm -o all stdout > $out_dir/result_permu.txt

    head -n 20 $out_dir/result_permu.txt

    # 4: Subscoring: subset utts into groups of different overlap rate
    # out_dir:
    #  |- ovlp_subscore:
    #         |- ref_0.20.trn
    #         |- ref_0.50.trn
    #             ...

done

for mode in "1mix"; do
    echo "Start $mode"
    decoding_path=$decode_root/${setname}_clean_${mode}
    out_dir=$decode_root/pi_scoring/${setname}_clean_${mode}
       
    log "Output dir: ${out_dir}"
    
    # 1: prepare output path
    mkdir -p $out_dir
    cp $0 $out_dir
    cp $decoding_path/score_wer/ref.trn $out_dir/ref_utt.trn
    cp $decoding_path/score_wer/hyp.trn $out_dir/hyp.trn
    ref_utt=$out_dir/ref_utt.trn
    hyp=$out_dir/hyp.trn
    
    rm -rf $out_dir/tmp
    rm -f $out_dir/{best_ref.trn,best_hyp.trn}
    
    # 2: splitting ref and hyp
    log "Splitting TRN."
    python3 search_best_permutation.py \
    		--ref_utt $ref_utt \
    		--hyp $hyp \
    		--out_dir $out_dir
    
    #- out_dir:
    #	- tmp:
    #	|	- utt0:
    #	|	|	- permu0 (utt-based)
    #	|	|	|	- ref.trn
    #	|	|	|	- hyp.trn
    #	|	|	|	- result.txt
    #	|	|	- permu1 (spkr-based)
    #	|	|	|	...
    #	|	|	- wers
    #	|	- utt1:
    #	|	|	...
    #	- best_ref.trn
    #	- best_hyp.trn
    
    # 3: scoring to get final error rate
    sclite \
    		-r $out_dir/best_ref.trn trn \
    		-h $out_dir/best_hyp.trn trn \
    		-i rm -o all stdout > $out_dir/result_permu.txt

    head -n 20 $out_dir/result_permu.txt
done


python get_res3.py --path ${decode_root}

log "Done."

