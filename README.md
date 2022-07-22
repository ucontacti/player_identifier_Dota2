![Actions Status](https://github.com/ucontacti/player_identifier_Dota2/workflows/CI/badge.svg)(https://github.com/ucontacti/player_identifier_Dota2/actions)

# player_identifier_Dota2
Player authentication in Dota 2 based on Keystroke and Mouse Movement

## Running
* Using Python(Make sure you are using python 3.8 or above):
    ```
    pip install -r requirements.txt  
    python app.py  
    ```

## Project explanation
The code consists of the following parts
### Replay Parser
Java code that uses [clarity](https://github.com/skadistats/clarity) to parse replays
### Pipeline
This pipeline is a Commmand-line script that lets you do these things:
#### player_id_scrapper
Scrape new players from Dotabuff
#### Pipeline
Update the pipeline by either adding new players to the pool or updating the existing ones
#### mouse_movement
Train a linear model using mouse movement data
#### itemization
Train a linear model for itemization features
### Results
<!-- The results of the authentication was over 90% and the detail report can be found [here](https://drive.google.com/open?id=1-332uLhMxQbwe6LelzNJXikF9g-gDSV-) -->

## Author
* **Saleh Daghigh** - [ucontacti](https://github.com/ucontacti)

This project was done under the supervision of Dr. Rafet Sifa from Fraunhofer Institute.