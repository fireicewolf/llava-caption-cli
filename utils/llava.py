import glob
import os
import time
from argparse import Namespace
from pathlib import Path

from PIL import Image
from llama_cpp import LLAMA_SPLIT_MODE_NONE, LLAMA_SPLIT_MODE_LAYER
from llama_cpp import Llama
from llama_cpp.llama_chat_format import Llava15ChatHandler, Llava16ChatHandler
from tqdm import tqdm

from utils.image import image_process, encode_image_to_base64
from utils.logger import Logger

SUPPORT_IMAGE_FORMATS = ("bmp", "jpg", "jpeg", "png")


class Llava:
    def __init__(
            self,
            logger: Logger,
            args: Namespace,
            base_model_path: Path,
            mmproj_model_path: Path,
            use_gpu: bool = False,
            chat_format: str = "llava-1-6"
    ):
        self.logger = logger
        self.args = args

        self.llm = None
        self.model_name = None
        self.base_model_path = base_model_path
        self.mmproj_model_path = mmproj_model_path
        self.use_gpu = use_gpu
        self.chat_format = chat_format

    def load_model(self):
        args = self.args
        self.model_name = model_name = args.model_name
        base_model_path = self.base_model_path
        mmproj_model_path = self.mmproj_model_path
        chat_format = self.chat_format
        use_gpu = True if self.use_gpu else False
        gpus = args.gpus
        n_ctx = args.n_ctx
        verbose = args.verbose
        chat_handler = Llava15ChatHandler(clip_model_path=str(mmproj_model_path), verbose=verbose) \
            if chat_format == "llava-1-5" \
            else Llava16ChatHandler(clip_model_path=str(mmproj_model_path), verbose=verbose)

        def calculating_split(gpu_count: int):
            from decimal import Decimal, getcontext
            if gpu_count > 1:
                getcontext().prec = 2
                value = Decimal(1) / gpu_count
                splited = [value] * (gpu_count - 1)
                splited.append(Decimal(1) - (Decimal(gpu_count) - 1) * value)
                return list(map(float, splited))
            else:
                return None

        split_in_gpus = [float(part.strip()) for part in args.split_in_gpus.split(",")] \
            if args.gpus > 1 else None
        split_in_gpus = calculating_split if split_in_gpus is None and gpus > 1 else split_in_gpus

        self.logger.info(f'Loading llm {model_name} with {"GPU" if use_gpu else "CPU"}...')
        start_time = time.monotonic()

        self.model_name = model_name
        self.llm = Llama(
            model_path=str(base_model_path),
            n_gpu_layers=-1 if use_gpu else 0,
            chat_format=chat_format,
            split_mode=LLAMA_SPLIT_MODE_NONE if gpus == 1 else LLAMA_SPLIT_MODE_LAYER,
            tensor_split=split_in_gpus if gpus > 1 and split_in_gpus is not None else None,
            chat_handler=chat_handler,
            n_ctx=n_ctx,  # n_ctx should be increased to accommodate the image embedding
            logits_all=True,
            verbose=verbose,
        )

        self.logger.info(f'{model_name} Loaded in {time.monotonic() - start_time:.1f}s.')

    def inference(self):
        args = self.args
        system_message = args.system_message
        user_prompt = args.user_prompt
        temp = args.temperature
        max_tokens = args.max_tokens
        datas_dir = args.data_path
        custom_caption_save_path = args.custom_caption_save_path
        recursive = args.recursive
        caption_extension = args.caption_extension
        not_overwrite = args.not_overwrite,
        image_size = args.image_size
        # Get image paths
        path_to_find = os.path.join(datas_dir, '**') if recursive else os.path.join(datas_dir, '*')
        image_paths = sorted(set(
            [image for image in glob.glob(path_to_find, recursive=recursive)
             if image.lower().endswith(SUPPORT_IMAGE_FORMATS)]),
            key=lambda filename: (os.path.splitext(filename)[0])
        ) if not os.path.isfile(datas_dir) else [str(datas_dir)] \
            if str(datas_dir).lower().endswith(SUPPORT_IMAGE_FORMATS) else None

        if image_paths is None:
            self.logger.error('Invalid dir or image path!')
            raise FileNotFoundError

        self.logger.info(f'Found {len(image_paths)} image(s).')

        def get_caption(
                llm: Llama,
                image: Image,
                system_message: str,
                user_prompt: str,
                temp: float,
                max_tokens: int = 40,
        ) -> str:
            if self.llm is None:
                self.logger.error("Llava model not loaded!!!")
                raise ReferenceError

            file_url = encode_image_to_base64(image)
            messages = [
                {"role": "system", "content": system_message},
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": file_url}},
                        {"type": "text", "text": user_prompt},
                    ],
                },
            ]

            start = time.monotonic()
            response = llm.create_chat_completion(
                messages=messages,
                temperature=temp,
                max_tokens=max_tokens,
            )
            self.logger.info(f"Response in {time.monotonic() - start:.1f}s")
            get_choice: dict = response["choices"][0]
            content = get_choice["message"]["content"]

            return content.strip()

        pbar = tqdm(total=len(image_paths), smoothing=0.0)
        for image_path in image_paths:
            try:
                pbar.set_description('Processing: {}'.format(image_path if len(image_path) <= 40 else
                                                             image_path[:15]) + ' ... ' + image_path[-20:])
                image = Image.open(image_path)
                image = image_process(image, image_size)
                caption = get_caption(
                    llm=self.llm,
                    image=image,
                    system_message=system_message,
                    user_prompt=user_prompt,
                    temp=temp,
                    max_tokens=max_tokens
                )
                if custom_caption_save_path is not None:
                    if not os.path.exists(custom_caption_save_path):
                        self.logger.error(f'{custom_caption_save_path} NOT FOUND!')
                        raise FileNotFoundError

                    self.logger.debug(f'Caption file(s) will be saved in {custom_caption_save_path}')

                    if os.path.isfile(datas_dir):
                        caption_file = str(os.path.splitext(os.path.basename(image_path))[0])

                    else:
                        caption_file = os.path.splitext(str(image_path)[len(str(datas_dir)):])[0]

                    caption_file = caption_file[1:] if caption_file[0] == '/' else caption_file
                    caption_file = os.path.join(args.custom_caption_save_path, caption_file)
                    # Make dir if not exist.
                    os.makedirs(os.path.join(str(caption_file)[:-len(os.path.basename(caption_file))]), exist_ok=True)
                    caption_file = Path(str(caption_file) + args.caption_extension)

                else:
                    caption_file = os.path.splitext(image_path)[0] + caption_extension

                if not_overwrite and os.path.isfile(caption_file):
                    self.logger.warning(f'Caption file {caption_file} already exist! Skip this caption.')
                    continue

                with open(caption_file, "wt", encoding="utf-8") as f:
                    f.write(caption + "\n")
                    self.logger.debug(f"\tImage path: {image_path}")
                    self.logger.debug(f"\tCaption path: {caption_file}")
                    self.logger.debug(f"\tCaption content: {caption}")

                pbar.update(1)
            except Exception as e:
                self.logger.error(f"Could not load image path: {image_path}, skip it.\nerror info: {e}")
                continue

        pbar.close()

    def unload_model(self):
        if self.llm is not None:
            self.logger.info(f'Unloading model {self.model_name}...')
            start = time.monotonic()
            del self.llm
            self.logger.info(f'{self.model_name} unloaded in {time.monotonic() - start:.1f}s.')
            del self.model_name
