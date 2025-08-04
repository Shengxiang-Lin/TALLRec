Our weights for the instruction tuning model is uploading [here](https://drive.google.com/file/d/1teUwLm4BOqhngfCKKXE1tiMhJPf_FvRJ/view?usp=sharing)

**TALLRec: An Effective and Efficient Tuning Framework to Align Large Language Model with Recommendation** is available at https://arxiv.org/abs/2305.00447.

**Wrongly delete the line in evaluate.py by mistake, now it has been updated**

We introduce a novel framework (TALLRec) that enables the efficient and effective adaptation of LLMs to recommendation tasks.

# Main results
Book     
|--------- | -----  | -----  | ----- | ----- | ----- | 
| Samples  | 16     | 32     | 64    | 128   | 256   |
|---------  | ----- | ----- | ----- | ----- | ----- | ----- |
| TALLRec  |       |        | 56.37  | 55.55 | 55.46 |
Movie       
|--------- | -----  | -----  | ----- | ----- | ----- | 
| Samples  | 16     | 32     | 64    | 128   | 256   |
|---------  | ----- | ----- | ----- | ----- | ----- | ----- |
| TALLRec  |        |        | 57.17 | 59.70 | 67.48 |


Table 1. we shown the AUC results of the frameworks on movie and book scenarios.

Train TALLRec base on LLaMA7B:
```
bash ./shell/instruct_7B.sh  gpu_id random_seed
```
If you want to run it under your environment, you need to make changes to the sh file:
- output_dir: Model save path，we will automatically add the seed and the sample to the end of the path for each experiments.
- base_model: LLaMA parameter weight path in Hugginface format
- train_data:  Training data path such as "./data/movie/train.json" for movie dataset.
- val_data: Validation data set path such as "./data/movie/valid.json" for movie dataset.
- instruction_model: The LoRA weights after the instruction tuning, for example lora weight from alpaca-lora.

After training, you need to evluate the test result on the best model evaluated by the validation set.
```
bash ./shell/evaluate.sh  gpu_id  output_dir
```
If you want to run it under your environment, you need to make changes to the sh file:
- base_model: LLaMA parameter weight path in Hugginface format
- test_data: Test data set path such as "./data/movie/test.json" for movie dataset.

Note that we will automatically detect all the different seed and sample files in the output_dir directory, and then integrate these results into the output_dir.json file.

Our project is developed based on the Alpaca_lora [repo](https://github.com/tloen/alpaca-lora), thanks for their contributions.

For "Environment setting sharing for CUDA 12.0", please see [here](https://github.com/SAI990323/TALLRec/issues/46).     

```
bash ./shell/instruct_7B/instruct_7B_book.sh  0 42
```

```
bash ./shell/evaluate.sh 1 results_medium_42_64/adapter_config.json 
```