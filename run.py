# Load creds modules
from helpers.handle_creds import (
    load_correct_creds
)

# Load helper modules
from helpers.parameters import (
    parse_args, load_config
)

from DeepQBot import DeepQBot

if __name__ == '__main__':
    
    # Load arguments then parse settings
    args = parse_args()

    # Load creds for correct environment
    DEFAULT_CREDS_FILE = 'creds.yml'
    creds_file = args.creds if args.creds else DEFAULT_CREDS_FILE
    parsed_creds = load_config(creds_file)
    access_key, secret_key = load_correct_creds(parsed_creds)

    # Create a new bot
    bot = DeepQBot(access_key, secret_key, CHANGE_IN_PRICE = 0.1)
    bot.trainNetwork()
