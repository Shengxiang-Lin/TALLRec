CUDA_ID=$1
model_dir=$2
model_name=$(basename "$model_dir")
result_json_data="evaluate_result/${model_name}_evaluate.json"
base_model='base_models/decapoda-research-llama-7B-hf'
test_data='data/book/test.json'
if [ ! -d "$model_dir" ]; then
    echo "The model directory $model_dir does not exist, creating it..."
    mkdir -p "$model_dir"
fi
result_dir=$(dirname "$result_json_data")
if [ ! -d "$result_dir" ]; then
    echo "The result directory $result_dir does not exist, creating it..."
    mkdir -p "$result_dir"
fi
echo "Evaluating model: $model_dir"
CUDA_VISIBLE_DEVICES=$CUDA_ID python evaluate.py \
    --base_model "$base_model" \
    --lora_weights "$model_dir" \
    --test_data_path "$test_data" \
    --result_json_data "$result_json_data"