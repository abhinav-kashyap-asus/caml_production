import bentoml
from rich.console import Console
from datasets import load_lookups, load_code_descriptions
import torch.nn as nn
import torch
from constants import MAX_LENGTH
from pydantic import BaseModel
from bentoml.io import JSON
from typing import Dict, Any
import json

VOCAB_DICTS_FILE = "../mimic3_vocab_dicts.json"
DESCRIPTIONS_JSON = "../code_descriptions.json"


class MedicalNote(BaseModel):
    note: str


with open(VOCAB_DICTS_FILE) as fp:
    dicts = json.load(fp)

with open(DESCRIPTIONS_JSON) as fp:
    code_descriptions = json.load(fp)


w2ind = dicts["w2ind"]
ind2c = dicts["ind2c"]

sigmoid = nn.Sigmoid()

console = Console()
runner = bentoml.pytorch.get("caml_pretrained_model").to_runner()
console.print("Load runner :white_check_mark:")

svc = bentoml.Service("caml_pretrained_model", runners=[runner])


@svc.api(
    input=JSON(pydantic_model=MedicalNote),
    output=JSON()
)
def bentoml_classify(note: MedicalNote) -> Dict[str, Any]:
    note_text = note.note
    # Split the text into tokens
    # OOV words are given a unique index at end of vocab lookup
    text = [int(w2ind[w]) if w in w2ind else len(w2ind) + 1 for w in note_text.split()]
    # truncate long documents
    if len(text) > MAX_LENGTH:
        text = text[: MAX_LENGTH]

    if len(text) < MAX_LENGTH:
        text.extend([0] * (MAX_LENGTH - len(text)))

    text = torch.LongTensor(text)

    # a single text has batch size of 1
    # add the 0th dimension
    text = text.unsqueeze(0)

    logits, _, _ = runner.run(text)
    logits = sigmoid(logits)

    # flatten the logits
    logits = logits.flatten()

    logits_sorted, indices = torch.sort(logits, descending=True)
    indices = indices.tolist()

    predicted_codes = [ind2c[str(idx)] for idx in indices[:10]]
    predicted_descriptions = [code_descriptions[predicted_code] for predicted_code in predicted_codes]

    return {
        "medical_codes": predicted_codes,
        "medical_descriptions": predicted_descriptions
    }
