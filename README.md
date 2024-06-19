# llava caption cli
A Python base cli tool for tagging images with llava models.

## Introduce

I make this repo because I want to caption some images cross-platform (On My old MBP, my game win pc or docker base linux cloud-server(like Google colab))

But I don't want to install a huge webui just for this little work. And some cloud-service are unfriendly to gradio base ui.

So this repo born.


## Model source

Huggingface are original sources, modelscope are pure forks from Huggingface(Because HuggingFace was blocked in Some place).

|        Model         |                                HuggingFace Link                                |                                      ModelScope Link                                       |
|:--------------------:|:------------------------------------------------------------------------------:|:------------------------------------------------------------------------------------------:|
| llava-v1.6-34B-gguf  |     [HuggingFace](https://huggingface.co/cjpais/llava-v1.6-34B-gguf)      |       [ModelScope](https://www.modelscope.cn/models/fireicewolf/llava-v1.6-34B-gguf)       |
| ggml_llava-v1.5-13b  |       [HuggingFace](https://huggingface.co/mys/ggml_llava-v1.5-13b)       |       [ModelScope](https://www.modelscope.cn/models/fireicewolf/ggml_llava-v1.5-13b)       |
|  ggml_llava-v1.5-7b  |    [HuggingFace](https://huggingface.co/mys/ggml_llava-v1.5-7b)     |      [ModelScope](https://www.modelscope.cn/models/fireicewolf/ggml_llava-v1.5-7b)      |

## TO-DO

make a simple ui by Jupyter widget(When my lazy cancer cured😊)

## Installation
Python 3.10 works fine. 

Open a shell terminal and follow below steps:
```shell
# Clone this repo
git clone https://github.com/fireicewolf/wd14-tagger-cli.git
cd wd14-tagger-cli

# create a Python venv
python -m venv .venv
.\venv\Scripts\activate

# Install dependencies
# Base dependencies, models for inference will download via python request libs.
pip install -U -r requirements.txt

# If you want to download or cache model via huggingface hub, install this.
pip install -U -r huggingface-requirements.txt

# If you want to download or cache model via modelscope hub, install this.
pip install -U -r modelscope-requirements.txt
```

### Take a notice
This project use llama-cpp-python as base lib, and it needs to be complied.

## Simple usage
__Make sure your python venv has been activated first!__
```shell
python caption.py your_datasets_path
```
To run with more options, You can find help by run with this or see at [Options](#options)
```shell
python caption.py -h
```

##  <span id="options">Options</span>
<details>
    <summary>Advance options</summary>
`data_path`

path for data

`--recursive`

Will include all support images format in your input datasets path and its sub-path.

`config`

config json for llava models, default is "default.json"

`--use_cpu`

Use cpu for inference.

`--gpus N`

how many gpus used for inference, default is 1.

`--split_in_gpus weights`

weights to split model in multi-gpus for inference. ex "0.5, 0.5" for 2 gpus balance.

`--n_ctx TEXT CONTEXT`

Text context, set it larger if your image is large, default is 2048.

`--model_name MODEL_NAME`

model name for inference, default is "llava-v1.6-34b.Q4_K_M", please check configs/default.json)

`--model_site MODEL_SITE`

Model site where onnx model download from(huggingface or modelscope), default is huggingface.

`--models_save_path MODEL_SAVE_PATH`

Path for models to save, default is models(under project folder).

`--download_method SDK`

Download models via sdk or url, default is sdk.

If huggingface hub or modelscope sdk not installed or download failed, will auto retry with url download.

`--use_sdk_cache`

Use huggingface or modelscope sdk cache to store models, this option need huggingface_hub or modelscope sdk installed.

If this enabled, `--models_save_path` will be ignored.

`--custom_model_path CUSTOM_MODEL_PATH`
`----custom_mmproj_path CUSTOM_MMPROJ_PATH`

This two args need to be used together. You can use your exist model.

`--custom_caption_save_path CUSTOM_CAPTION_SAVE_PATH`

Save caption files to a custom path but not with images(But keep their directory structure)

`--log_level LOG_LEVEL`

Log level for terminal console and log file, default is `INFO`(`DEBUG`,`INFO`,`WARNING`,`ERROR`,`CRITICAL`)

`--save_logs`

Save logs to a file, log will be saved at same level with `data_dir_path`

`--caption_extension CAPTION_EXTENSION`

Caption file extension, default is `.txt`

`--not_overwrite`

Do not overwrite caption file if it existed.

`--system_message SYSTEM_MESSAGE`

system message for llava model.

`--user_prompt USER_PROMPT`

user prompt for caption.

`--temperature TEMPERATURE`

temperature for llava model,default is 0.4.

`--max_tokens MAX_TOKENS`

max tokens for output.

`--verbose`

llama-cpp-python verbose mode.

</details>

## Credits
Base on [llama-cpp-python](https://github.com/abetlen/llama-cpp-python)

Without their works(👏👏), this repo won't exist.