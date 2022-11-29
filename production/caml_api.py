from fastapi import FastAPI
from pydantic import BaseModel
from model_infer import Infer
from get_model import get_model
from datasets import load_lookups

app = FastAPI()

# DECLARE SOME CONSTANTS THAT IS HARDCODED HERE FOR DEMO PURPOSES
label_space = "full"
embed_file = "/Users/abhinavkashyap/abhi/projects/icd_coding/data/mimic3_caml/processed_full.embed"
filter_size = 10
num_filter_maps = 50
_lambda = 0
gpu = False
public_model = True
vocab_file = "/Users/abhinavkashyap/abhi/projects/icd_coding/data/mimic3_caml/vocab.csv"
version = "mimic3"
model_name = "conv_attn"
data_path = "/Users/abhinavkashyap/abhi/projects/icd_coding/data/mimic3_caml/train_full.csv"
model_path = "/Users/abhinavkashyap/abhi/projects/icd_coding/data/caml_reproduction_model/model_best_prec_at_8.pth"

args = dict(
    public_model=public_model,
    version=version,
    vocab=vocab_file,
    model=model_name,
    Y=label_space,
    data_path=data_path
)

dicts = load_lookups(args)

model = get_model(
    model_path,
    label_space,
    embed_file,
    filter_size,
    num_filter_maps,
    _lambda,
    gpu,
    public_model,
    vocab_file,
    version,
    model_name,
    data_path
)

infer = Infer(model=model, dicts=dicts)


class MedicalNote(BaseModel):
    note: str


@app.get("/")
def home():
    return {
        "message": "CAML Model Predictions"
    }


@app.post("/caml_predict/")
def caml_predict(note: MedicalNote):
    labels, descriptions = infer.predict(note_text=note.note)
    return {
        "medical_codes": labels,
        "medical_descriptions": descriptions
    }
