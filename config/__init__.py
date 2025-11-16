import os
from dotenv import load_dotenv

if not os.getenv("_ENV_LOADED"):
    load_dotenv()
    os.environ["_ENV_LOADED"] = "1"
