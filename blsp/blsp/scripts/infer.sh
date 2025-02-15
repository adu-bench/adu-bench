export PYTHONPATH="./:$PYTHONPATH"
export TORCH_HOME="/your/path"

CUDA_VISIBLE_DEVICES=0 python3 blsp/generate.py \
    --input_file "/your/path/input_file.jsonl" \
    --output_file "/your/path/output_file.jsonl" \
    --blsp_model "/your/path/blsp_lslm_7b" \
    --instruction ""
