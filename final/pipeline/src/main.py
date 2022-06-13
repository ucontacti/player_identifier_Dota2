import typer
from os.path import exists

from pipeline.pipeline_handler import add_replay_to_pipeline, \
                                      update_pipeline, \
                                      create_empty_replay_tracker
from pipeline.player_id_scraper import scrape
from model.classifier import Classifier

REPLAY_TRACKER_PATH = "../resources/replay_tracker.csv"

app = typer.Typer()

@app.command()
def player_id_scrapper(driver_path: str = typer.Option(..., "--driver_path", "-d"), num_player: int = typer.Option(10, "--num_player", "-n", show_default=True)):
    if not exists(driver_path):
        typer.echo("chromedriver not found!")
    else:
        scrape(driver_path, num_player)


@app.command()
def pipeline(fetch_new: bool = False):
    if not exists(REPLAY_TRACKER_PATH):
        create_empty_replay_tracker()
    if fetch_new:
        typer.echo("adding new replays to pipline.")
        add_replay_to_pipeline()
    update_pipeline()

@app.command()
def model(model_num: int =typer.Option(1, prompt="What is the name of the model?\n1) Logistic Regression\n2) Random Forest\n3) Decision Tree\nYour choice"),\
        show_default=True):
    model = Classifier()
    model.select_model(model_num)
    model.train_and_eval()
    sure = typer.confirm("Do you want to calculate coefficiency?")
    if sure:
        model.calculate_coefficiency()
    sure = typer.confirm("Do you want to plot results?")
    if sure:
        model.plot_data()
    sure = typer.confirm("Do you want to save model?")
    if sure:
        model.save_model()    

if __name__ == "__main__":
    app()