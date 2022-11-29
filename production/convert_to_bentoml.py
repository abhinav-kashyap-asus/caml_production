from rich.console import Console
import bentoml
import click
from get_model import get_model


@click.command()
@click.option("--vocab-dicts-file", type=click.Path(exists=True, file_okay=True, dir_okay=False, readable=True),
              required=False,
              help="path to a file holding pre-trained embeddings")
@click.option("--filter-size", type=int, required=True, help="The size of the kernel used for convolution")
@click.option("--num-filter-maps", type=int, required=True, help="The number of output filter maps in the convolution")
@click.option("--gpu/--no-gpu", default=False, help="Optional flag to use GPU if available")
@click.option("--model-path", type=click.Path(exists=True, file_okay=True, dir_okay=False, readable=True),
              required=True,
              help="Path where the model parameters are stored")
def main(
        vocab_dicts_file,
        filter_size,
        num_filter_maps,
        gpu,
        model_path
):
    console = Console()

    model = get_model(
        vocab_dicts_file,
        model_path,
        filter_size,
        num_filter_maps,
        0,
        gpu,
    )

    console.print("Load Pytorch model :white_check_mark:")

    with console.status(f"Converting the model to bento ml"):
        bentoml.pytorch.save_model(
            name="caml_pretrained_model",
            model=model,
            signatures={"__call__": {"batchable": False, "batch_dim": 0}},
        )

    console.print("Add pytorch model to bentoml repository :white_check_mark:")


if __name__ == "__main__":
    main()
