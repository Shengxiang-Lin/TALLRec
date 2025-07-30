import sys
import fire
import gradio as gr
import torch
torch.set_num_threads(1)
import transformers
import json
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
from peft import PeftModel
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer
from sklearn.metrics import roc_auc_score

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

try:
    if torch.backends.mps.is_available():
        device = "mps"
except:  # noqa: E722
    pass

def main(
    load_8bit: bool = False,
    base_model: str = "base_models/decapoda-research-llama-7B-hf",
    lora_weights: str = "results_medium/results_medium_book_42_64",
    test_data_path: str = "data/book/test-1.json",
    result_json_data: str = "tmp.json",
    batch_size: int = 2,
    share_gradio: bool = False,
):
    assert base_model, "Please specify a --base_model"

    model_type = lora_weights.split('/')[-1]
    model_name = '_'.join(model_type.split('_')[:2])

    train_sce = 'book' if 'book' in model_type else 'movie'
    test_sce = 'book' if 'book' in test_data_path else 'movie'
    temp_list = model_type.split('_')
    seed = temp_list[-2]
    sample = temp_list[-1]

    if os.path.exists(result_json_data):
        with open(result_json_data, 'r') as f:
            data = json.load(f)
    else:
        data = dict()

    data.setdefault(train_sce, {}).setdefault(test_sce, {}).setdefault(model_name, {}).setdefault(seed, {})
    if sample in data[train_sce][test_sce][model_name][seed]:
        exit(0)

    tokenizer = LlamaTokenizer.from_pretrained(base_model)
    if device == "cuda":
        model = LlamaForCausalLM.from_pretrained(
            base_model,
            load_in_8bit=load_8bit,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            torch_dtype=torch.float16,
            device_map={'': 0},
            local_files_only=True
        )
    elif device == "mps":
        model = LlamaForCausalLM.from_pretrained(
            base_model,
            device_map={"": device},
            torch_dtype=torch.float16,
        )
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            device_map={"": device},
            torch_dtype=torch.float16,
            local_files_only=True
        )
    else:
        model = LlamaForCausalLM.from_pretrained(
            base_model, device_map={"": device}, low_cpu_mem_usage=True
        )
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            device_map={"": device},
            local_files_only=True
        )

    tokenizer.padding_side = "left"
    model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
    model.config.bos_token_id = 1
    model.config.eos_token_id = 2

    if not load_8bit:
        model.half()

    model.eval()
    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    def evaluate(
        instructions,
        inputs=None,
        temperature=0,
        top_p=1.0,
        top_k=40,
        num_beams=1,
        max_new_tokens=16,
        batch_size=1,
        **kwargs,
    ):
        prompt = [generate_prompt(instruction, input) for instruction, input in zip(instructions, inputs)]
        inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)
        generation_config = GenerationConfig(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            num_beams=num_beams,
            **kwargs,
        )
        with torch.no_grad():
            generation_output = model.generate(
                **inputs,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=max_new_tokens,
            )
        s = generation_output.sequences
        scores = generation_output.scores[0].softmax(dim=-1)
        logits = scores[:, [8241, 3782]].clone().detach().to(torch.float32).softmax(dim=-1)
        output = tokenizer.batch_decode(s, skip_special_tokens=True)
        output = [_.split('Response:\n')[-1] for _ in output]
        return output, logits.tolist()

    outputs = []
    logits = []
    gold = []
    pred = []

    with open(test_data_path, 'r') as f:
        test_data = json.load(f)
        instructions = [_['instruction'] for _ in test_data]
        inputs = [_['input'] for _ in test_data]
        gold = [int(_['output'] == 'Yes.') for _ in test_data]

        def batch(lst, batch_size=32):
            chunk_size = (len(lst) - 1) // batch_size + 1
            for i in range(chunk_size):
                yield lst[batch_size * i: batch_size * (i + 1)]

        from tqdm import tqdm
        for i, batch_pack in tqdm(enumerate(zip(batch(instructions), batch(inputs)))):
            instructions_batch, inputs_batch = batch_pack
            output, logit = evaluate(instructions_batch, inputs_batch)
            outputs += output
            logits += logit

        for i, test in enumerate(test_data):
            test_data[i]['predict'] = outputs[i]
            test_data[i]['logits'] = logits[i]
            pred.append(logits[i][0])  # 取Yes的概率作为打分

    # 打印前5个样本的预测信息
    print("\n==== Top 5 Predictions ====")
    for i in range(min(5, len(test_data))):
        print(f"[{i}] Instruction: {test_data[i]['instruction']}")
        print(f"    Input: {test_data[i]['input']}")
        print(f"    Predict: {test_data[i]['predict']}")
        print(f"    Logits (Yes, No): {test_data[i]['logits']}")
        print(f"    Gold: {'Yes.' if gold[i] == 1 else 'No.'}")
        print("-" * 50)

    from sklearn.metrics import roc_auc_score
    auc = roc_auc_score(gold, pred)
    data[train_sce][test_sce][model_name][seed][sample] = auc

    # 保存AUC结果
    with open(result_json_data, 'w') as f:
        json.dump(data, f, indent=4)

    # 保存所有样本预测详情
    with open("detailed_predictions.json", "w") as f_pred:
        json.dump(test_data, f_pred, indent=4, ensure_ascii=False)

def generate_prompt(instruction, input=None):
    if input:
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input}

### Response:
"""
    else:
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:
"""

if __name__ == "__main__":
    fire.Fire(main)
