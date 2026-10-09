import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration


class RoadSceneVLM:

    def __init__(
        self,
        model_name="Salesforce/blip-image-captioning-base"
    ):
        print("=" * 50)
        print("           INITIALIZING VLM")
        print("=" * 50)

        if torch.cuda.is_available():
            self.device = torch.device("cuda")

            print("VLM device :", self.device)
            print(
                "GPU        :",
                torch.cuda.get_device_name(0)
            )
        else:
            self.device = torch.device("cpu")
            print("VLM device : CPU")

        print("Model      :", model_name)
        print("=" * 50)

        self.processor = BlipProcessor.from_pretrained(
            model_name
        )

        if self.device.type == "cuda":
            self.model = (
                BlipForConditionalGeneration
                .from_pretrained(
                    model_name,
                    torch_dtype=torch.float16
                )
            )
        else:
            self.model = (
                BlipForConditionalGeneration
                .from_pretrained(
                    model_name
                )
            )

        self.model.to(self.device)
        self.model.eval()

        print("VLM loaded successfully!")
        print()

    def generate_caption(
        self,
        image_path,
        max_new_tokens=50
    ):
        """
        Generate a visual description using BLIP.

        BLIP is used only for visual understanding.
        Structured object/risk information is handled
        separately by YOLO11 and the scene graph.
        """

        image = Image.open(
            image_path
        ).convert("RGB")

        # IMPORTANT:
        # Do not use the previous long prompt.
        # BLIP captioning works better with a simple
        # image-captioning request.
        prompt = "a photo of"

        inputs = self.processor(
            images=image,
            text=prompt,
            return_tensors="pt"
        )

        inputs = {
            key: value.to(self.device)
            for key, value in inputs.items()
        }

        with torch.no_grad():

            if self.device.type == "cuda":

                with torch.autocast(
                    device_type="cuda",
                    dtype=torch.float16
                ):
                    output = self.model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        num_beams=3,
                        repetition_penalty=1.2,
                        no_repeat_ngram_size=3
                    )

            else:

                output = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    num_beams=3,
                    repetition_penalty=1.2,
                    no_repeat_ngram_size=3
                )

        caption = self.processor.decode(
            output[0],
            skip_special_tokens=True
        )

        caption = caption.strip()

        # Remove prompt if BLIP repeats it
        if caption.lower().startswith("a photo of"):
            caption = caption[10:].strip()

        return caption