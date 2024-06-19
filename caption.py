import argparse
import json
import os
from datetime import datetime
from pathlib import Path

from utils.download import download
from utils.llava import Llava
from utils.logger import Logger


DEFAULT_SYSTEM_MESSAGE = """
You are an assistant who describes the content and composition of images.
Describe only what you see in the image, not what you think the image is about.
Be factual and literal. Do not use metaphors or similes. Be concise.
"""
DEFAULT_USER_PROMPT = """
Please describe this image in 30 to 50 words.
"""


def main(args):
    # Set logger
    workspace_path = os.getcwd()
    data_dir_path = Path(args.data_path)
    log_file_path = data_dir_path.parent if os.path.exists(data_dir_path.parent) else workspace_path

    log_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    # caption_failed_list_file = f'Caption_failed_list_{log_time}.txt'

    if os.path.exists(data_dir_path):
        log_name = os.path.basename(data_dir_path)

    else:
        print(f'{data_dir_path} NOT FOUND!!!')
        raise FileNotFoundError

    if args.save_logs:
        log_file = f'Caption_{log_name}_{log_time}.log' if log_name else f'test_{log_time}.log'
        log_file = os.path.join(log_file_path, log_file) \
            if os.path.exists(log_file_path) else os.path.join(os.getcwd(), log_file)
    else:
        log_file = None

    if str(args.log_level).lower() in 'debug, info, warning, error, critical':
        my_logger = Logger(args.log_level, log_file).logger
        my_logger.info(f'Set log level to "{args.log_level}"')

    else:
        my_logger = Logger('INFO', log_file).logger
        my_logger.warning('Invalid log level, set log level to "INFO"!')

    if args.save_logs:
        my_logger.info(f'Log file will be saved as "{log_file}".')

    # Check custom models path
    config_file = os.path.join(Path(__file__).parent, 'configs', 'default.json') \
        if args.config == "default.json" else Path(args.config)

    if args.custom_model_path is not None and args.custom_mmproj_path is not None:
        # Use custom model and mmproj path
        my_logger.warning('custom_model_path and custom_mmproj_path are enabled')
        if not (os.path.isfile(args.custom_model_path) and str(args.custom_model_path).endswith('.gguf')):
            my_logger.error(f'{args.custom_model_path} is not a gguf file!')
            raise FileNotFoundError

        elif not (os.path.isfile(args.custom_mmproj_path) and str(args.custom_mmproj_path).endswith('.gguf')):
            my_logger.error(f'{args.custom_mmproj_path} is not a gguf file!')
            raise FileNotFoundError

        model_path, mmproj_path = args.custom_model_path, args.custom_mmproj_path

    else:
        if args.custom_model_path is not None and args.custom_mmproj_path is None:
            my_logger.warning(f'custom_model_path has been set, but custom_mmproj_path not set. '
                              f'Will ignore these setting!')
        elif args.custom_model_path is None and args.custom_mmproj_path is not None:
            my_logger.warning(f'custom_mmproj_path has been set, but custom_model_path not set. '
                              f'Will ignore these setting!')

        # Download llava model and mmproj
        if os.path.exists(Path(args.models_save_path)):
            models_save_path = Path(args.models_save_path)

        else:
            models_save_path = Path(os.path.join(Path(__file__).parent, args.models_save_path))

        model_path, mmproj_path = download(
            logger=my_logger,
            config_file=config_file,
            model_name=str(args.model_name),
            model_site=str(args.model_site),
            models_save_path=models_save_path,
            use_sdk_cache=True if args.use_sdk_cache else False,
            download_method=str(args.download_method)
        )

    # Load models
    model_name = args.model_name
    with open(config_file, 'r', encoding='utf-8') as config_json:
        datas = json.load(config_json)
    chat_format = datas[model_name]["chat_format"]

    my_llava = Llava(
        logger=my_logger,
        args=args,
        base_model_path=model_path,
        mmproj_model_path=mmproj_path,
        use_gpu=False if args.use_cpu else True,
        chat_format=chat_format
    )
    my_llava.load_model()

    # Inference
    my_llava.inference()

    # Unload models
    my_llava.unload_model()


def setup_args() -> argparse.ArgumentParser:
    args = argparse.ArgumentParser()

    args.add_argument(
        'data_path',
        type=str,
        help='path for data.'
    )
    args.add_argument(
        '--recursive',
        action='store_true',
        help='Include recursive dirs'
    )
    args.add_argument(
        '--config',
        type=str,
        default='default.json',
        help='config json for llava models, default is "default.json"'
    )
    args.add_argument(
        '--use_cpu',
        action='store_true',
        help='use cpu for inference.'
    )
    args.add_argument(
        '--gpus',
        type=int,
        default=1,
        help='how many gpus used for inference, default is 1'
    )
    args.add_argument(
        '--split_in_gpus',
        type=str,
        help='weights to split model in multi-gpus for inference.'
    )
    args.add_argument(
        '--n_ctx',
        type=int,
        default=2048,
        help='Text context, set it larger if your image is large, default is 2048.'
    )
    args.add_argument(
        '--image_size',
        type=int,
        default=1024,
        help='resize image to suitable, default is 1024.'
    )
    args.add_argument(
        '--model_name',
        type=str,
        default='llava-v1.6-34b.Q4_K_M',
        help='model name for inference, default is "llava-v1.6-34b.Q4_K_M", please check configs/default.json'
    )
    args.add_argument(
        '--model_site',
        type=str,
        choices=['huggingface', 'modelscope'],
        default='huggingface',
        help='download model from model site huggingface or modelscope, default is "huggingface".'
    )
    args.add_argument(
        '--models_save_path',
        type=str,
        default="models",
        help='path to save models, default is "models".'
    )
    args.add_argument(
        '--use_sdk_cache',
        action='store_true',
        help='use sdk\'s cache dir to store models. \
            if this option enabled, "--models_save_path" will be ignored.'
    )
    args.add_argument(
        '--download_method',
        type=str,
        choices=["SDK", "URL"],
        default='SDK',
        help='download method via SDK or URL, default is "SDK".'
    )
    args.add_argument(
        '--custom_model_path',
        type=str,
        default=None,
        help='Input custom base model path, you should use --custom_mmproj_path together, '
             'otherwise this will be ignored'
    )
    args.add_argument(
        '--custom_mmproj_path',
        type=str,
        default=None,
        help='Input custom mmproj model path, you should use --custom_model_path together, otherwise this will be '
             'ignored'
    )
    args.add_argument(
        '--custom_caption_save_path',
        type=str,
        default=None,
        help='Input custom caption file save path.'
    )
    args.add_argument(
        '--log_level',
        type=str,
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        default='INFO',
        help='set log level, default is "INFO"'
    )
    args.add_argument(
        '--save_logs',
        action='store_true',
        help='save log file.'
    )
    args.add_argument(
        '--caption_extension',
        type=str,
        default='.txt',
        help='extension of caption file, default is ".txt"'
    )
    args.add_argument(
        '--not_overwrite',
        action='store_true',
        help='not overwrite caption file if exist.'
    )
    args.add_argument(
        '--system_message',
        type=str,
        default=DEFAULT_SYSTEM_MESSAGE,
        help='system message for llava model.'
    )
    args.add_argument(
        '--user_prompt',
        type=str,
        default=DEFAULT_USER_PROMPT,
        help='user prompt for caption.'
    )
    args.add_argument(
        '--temperature',
        type=float,
        default=0.4,
        help='temperature for llava model.'
    )
    args.add_argument(
        '--max_tokens',
        type=int,
        default=40,
        help='max tokens for output.'
    )
    args.add_argument(
        '--verbose',
        action='store_true',
        help='llama-cpp-python verbose mode.'
    )

    return args


if __name__ == "__main__":
    args = setup_args()
    args = args.parse_args()
    main(args)
