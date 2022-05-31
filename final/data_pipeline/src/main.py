import typer
from os.path import exists

from pipeline.pipeline_handler import add_replay_to_pipeline, \
                                      update_pipeline, \
                                      create_empty_replay_tracker
from pipeline.player_id_scraper import scrape

REPLAY_TRACKER_PATH = "../resources/replay_tracker.csv"


app = typer.Typer()

@app.command()
def player_id_scrapper(driver_path: str, num_player: int = 10):
    if not exists(driver_path):
        typer.echo("chromedriver not found!")
    else:
        scrape(driver_path, num_player)



@app.command()
def pipeline(fetch_new: bool = False):
    if fetch_new:
        typer.echo("adding new replays to pipline.")
        add_replay_to_pipeline()
    if not exists(REPLAY_TRACKER_PATH):
        create_empty_replay_tracker()
    update_pipeline()
    

if __name__ == "__main__":
    app()