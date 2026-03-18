from pathlib import Path

import torch
from transformers import AutoModel, AutoTokenizer


MODEL_NAME = "sergeyzh/rubert-mini-frida"
OUTPUT_PATH = Path("models/onnx/model.onnx")


def main() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME)
    model.eval()

    sample = tokenizer(
        ["пример текста"],
        padding=True,
        truncation=True,
        max_length=32,
        return_tensors="pt",
    )

    input_names = ["input_ids", "attention_mask"]
    args = (sample["input_ids"], sample["attention_mask"])

    if "token_type_ids" in sample:
        input_names.append("token_type_ids")
        args = (sample["input_ids"], sample["attention_mask"], sample["token_type_ids"])

    dynamic_axes = {
        "input_ids": {0: "batch", 1: "sequence"},
        "attention_mask": {0: "batch", 1: "sequence"},
        "last_hidden_state": {0: "batch", 1: "sequence"},
    }
    if "token_type_ids" in sample:
        dynamic_axes["token_type_ids"] = {0: "batch", 1: "sequence"}

    with torch.no_grad():
        torch.onnx.export(
            model,
            args=args,
            f=str(OUTPUT_PATH),
            input_names=input_names,
            output_names=["last_hidden_state"],
            dynamic_axes=dynamic_axes,
            opset_version=17,
            do_constant_folding=True,
        )

    print(f"ONNX model exported to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
