#!/usr/bin/env python3

import os, sys
import uvicorn
import configparser
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from celery import Celery
from dotenv import load_dotenv

# Import route modules
from api_routes import knowledge, tools, system, core, auth, uspto, feedback
from api_routes.models import *

# Import existing components
from sources.agents.general_agent import GeneralAgent
from sources.llm_provider import Provider
from sources.interaction import Interaction
from sources.browser import Browser, create_driver
from sources.utility import pretty_print
from sources.logger import Logger

load_dotenv()

def is_running_in_docker():
    """Detect if code is running inside a Docker container."""
    # Method 1: Check for .dockerenv file
    if os.path.exists('/.dockerenv'):
        return True
    
    # Method 2: Check cgroup
    try:
        with open('/proc/1/cgroup', 'r') as f:
            return 'docker' in f.read()
    except:
        pass
    
    return False

def initialize_system():
    config = configparser.ConfigParser()
    config.read('config.ini')
    
    stealth_mode = config.getboolean('BROWSER', 'stealth_mode')
    personality_folder = "jarvis" if config.getboolean('MAIN', 'jarvis_personality') else "base"
    languages = config["MAIN"]["languages"].split(' ')
    
    # Force headless mode in Docker containers
    headless = config.getboolean('BROWSER', 'headless_browser')
    if is_running_in_docker() and not headless:
        # Print prominent warning to console (visible in docker-compose output)
        print("\n" + "*" * 70)
        print("*** WARNING: Detected Docker environment - forcing headless_browser=True ***")
        print("*** INFO: To see the browser, run 'python cli.py' on your host machine ***")
        print("*" * 70 + "\n")
        
        # Flush to ensure it's displayed immediately
        sys.stdout.flush()
        
        # Also log to file
        logger.warning("Detected Docker environment - forcing headless_browser=True")
        logger.info("To see the browser, run 'python cli.py' on your host machine instead")
        
        headless = True
    
    provider = Provider(
        provider_name=config["MAIN"]["provider_name"],
        model=config["MAIN"]["provider_model"],
        server_address=config["MAIN"]["provider_server_address"],
        is_local=config.getboolean('MAIN', 'is_local')
    )
    logger.info(f"Provider initialized: {provider.provider_name} ({provider.model})")

    browser = Browser(
        create_driver(headless=headless, stealth_mode=stealth_mode, lang=languages[0]),
        anticaptcha_manual_install=stealth_mode
    )
    logger.info("Browser initialized")

    agents = [
        GeneralAgent(
            name="General",
            prompt_path=f"prompts/{personality_folder}/general_agent.txt",
            provider=provider, verbose=False
        )
    ]
    logger.info("Agents initialized")

    interaction = Interaction(
        agents,
        tts_enabled=config.getboolean('MAIN', 'speak'),
        stt_enabled=config.getboolean('MAIN', 'listen'),
        recover_last_session=config.getboolean('MAIN', 'recover_last_session'),
        langs=languages
    )
    logger.info("Interaction initialized")
    return interaction, config

# Initialize FastAPI app
api = FastAPI(title="CopiioAI API", version="0.1.0")

# Initialize Celery
celery_app = Celery("tasks", broker="redis://localhost:6379/0", backend="redis://localhost:6379/0")
celery_app.conf.update(task_track_started=True)

# Initialize logger
logger = Logger("backend.log")

# Add CORS middleware
api.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
if not os.path.exists(".screenshots"):
    os.makedirs(".screenshots")
api.mount("/screenshots", StaticFiles(directory=".screenshots"), name="screenshots")

# Initialize system
interaction, config = initialize_system()
is_generating = False
query_resp_history = []

# Helper function for query processing
async def think_wrapper(user_id, interaction, query, query_id):
    try:
        success = False
        interaction.last_query = query
        interaction.query_id = query_id
        logger.info("Agents request is being processed")
        success = await interaction.think(user_id)
        if not success:
            interaction.last_answer = "Error: No answer from agent"
            interaction.last_reasoning = "Error: No reasoning from agent"
            interaction.last_success = False
        else:
            interaction.last_success = True
        pretty_print(interaction.last_answer)
        interaction.speak_answer()
        return success
    except Exception as e:
        logger.error(f"Error in think_wrapper: {str(e)}")
        interaction.last_answer = f""
        interaction.last_reasoning = f"Error: {str(e)}"
        interaction.last_success = False
        raise e

async def create_agent():
    provider = Provider(
        provider_name=config["MAIN"]["provider_name"],
        model=config["MAIN"]["provider_model"],
        server_address=config["MAIN"]["provider_server_address"],
        is_local=config.getboolean('MAIN', 'is_local')
    )
    return GeneralAgent(
        name="General",
        prompt_path=f"prompts/jarvis/general_agent.txt",
        provider=provider, verbose=False
    )

# Include route modules (without prefix to maintain original API structure)
api.include_router(knowledge.router, tags=["knowledge"])
api.include_router(tools.router, tags=["tools"])
api.include_router(auth.router, tags=["auth"])
api.include_router(uspto.router, tags=["uspto"])
system_router = system.register_system_routes(logger, interaction, query_resp_history, config)
api.include_router(system_router, tags=["system"])
core_router = core.register_core_routes(logger, interaction, query_resp_history, config, is_generating, think_wrapper, create_agent)
api.include_router(core_router, tags=["core"])
feedback_router = feedback.register_feedback_routes(logger, config)
api.include_router(feedback_router, tags=["feedback"])
# Note: query router is not included as it contained conflicting endpoints and is now empty

if __name__ == "__main__":
    # Print startup info
    if is_running_in_docker():
        print("[CopiioAI] Starting in Docker container...")
    else:
        print("[CopiioAI] Starting on host machine...")
    
    envport = os.getenv("BACKEND_PORT")
    if envport:
        port = int(envport)
    else:
        port = 7777
    
    uvicorn.run(api, host="0.0.0.0", port=port)
