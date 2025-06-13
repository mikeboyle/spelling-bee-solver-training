from bs4 import BeautifulSoup
import requests
import re
import json
from datetime import datetime
from typing import Any
from src.constants import DATE_FORMAT, SPELLING_BEE_URL
from src.fileutils import dump_json_to_file

def fetch_today_puzzle() -> dict[str, Any]:
    """Fetches the current day's Spelling Bee puzzle and loads as JSON object"""
    res = requests.get(SPELLING_BEE_URL)
    soup = BeautifulSoup(res.text, "html.parser")
    script = soup.find("script", string=re.compile("window.gameData"))
    if script:
        script_text = script.get_text()
        offset = len("window.gameData = ")
        script_data = json.loads(script_text[offset:])
        today_puzzle = script_data['today']
        return today_puzzle
    else:
        raise ValueError("<script> tag with winodw.gameData not found!")

def write_puzzle_to_file(puzzle: dict[str, Any]) -> None:
    """Writes a puzzle to a YYYY-MM-DD.json file in the year=YYYY/month=MM/ path for its date"""
    date_str = puzzle["printDate"]
    date = datetime.strptime(date_str, DATE_FORMAT)
    
    filename = f"{date_str}.json"
    file_path = f"raw/solutions/year={date.strftime('%Y')}/month={date.strftime('%m')}/{filename}"

    print(f"saving to {file_path}...")
    dump_json_to_file(puzzle, file_path, sort_keys=False)
    
def run() -> None:
    """Checks for the latest puzzle and saves it to file"""
    print("Checking spelling bee for latest puzzle...")
    today_puzzle = fetch_today_puzzle()
    write_puzzle_to_file(today_puzzle)
    