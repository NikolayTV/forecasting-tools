from __future__ import annotations

import argparse
import asyncio
import os
import sys

import dotenv

# Dynamically determine the absolute path to the top-level directory
current_dir = os.path.dirname(os.path.abspath(__file__))
top_level_dir = os.path.abspath(os.path.join(current_dir, "../"))
sys.path.append(top_level_dir)
dotenv.load_dotenv()

import logging

from forecasting_tools.forecasting.forecast_bots.main_bot import MainBot
from forecasting_tools.forecasting.forecast_bots.template_bot import (
    TemplateBot,
)
from forecasting_tools.forecasting.forecast_bots.template_v1_bot import (
    TemplateBot_v1,
)
from forecasting_tools.forecasting.helpers.forecast_database_manager import (
    ForecastDatabaseManager,
    ForecastRunType,
)
from forecasting_tools.forecasting.helpers.metaculus_api import MetaculusApi
from forecasting_tools.util.custom_logger import CustomLogger

logger = logging.getLogger(__name__)


def get_forecaster(bot_type: str, allow_rerun: bool) -> TemplateBot | MainBot | TemplateBot_v1:
    # Print environment variables for debugging
    env_vars = [
        "OPENAI_API_KEY",
        "METACULUS_TOKEN",
        "ANTHROPIC_API_KEY",
        "PERPLEXITY_API_KEY",
        "EXA_API_KEY",
        "CODA_API_KEY",
        "HUGGINGFACE_API_KEY"
    ]

    print("\nEnvironment variables status:")
    for var in env_vars:
        value = os.getenv(var)
        is_set = "✓" if value else "✗"
        # Only show first/last 4 chars if value exists
        display_value = f"{value[:4]}...{value[-4:]}" if value else "Not set"
        print(f"{var}: {is_set} {display_value}")
    print()

    file_path = "logs/forecasts/forecast_bot/"
    skip_previously_forecasted_questions = not allow_rerun
    if bot_type == "template_v1":
        return TemplateBot_v1(
            research_reports_per_question=5,
            predictions_per_research_report=2,
            use_research_summary_to_forecast=True,
            publish_reports_to_metaculus=True,
            folder_to_save_reports_to=file_path,
            skip_previously_forecasted_questions=skip_previously_forecasted_questions,
        )
    elif bot_type == "template":
        return TemplateBot(
            research_reports_per_question=3,
            predictions_per_research_report=3,
            publish_reports_to_metaculus=True,
            folder_to_save_reports_to=file_path,
            skip_previously_forecasted_questions=skip_previously_forecasted_questions,
        )
    else:
        return MainBot(
            publish_reports_to_metaculus=True,
            folder_to_save_reports_to=file_path,
            skip_previously_forecasted_questions=skip_previously_forecasted_questions,
        )


async def run_morning_forecasts(bot_type: str, allow_rerun: bool) -> None:
    CustomLogger.setup_logging()
    forecaster = get_forecaster(bot_type, allow_rerun)
    TOURNAMENT_ID = MetaculusApi.AI_COMPETITION_ID_Q1
    # TOURNAMENT_ID = MetaculusApi.AI_WARMUP_TOURNAMENT_ID
    reports = await forecaster.forecast_on_tournament(TOURNAMENT_ID)

    if os.environ.get("CODA_API_KEY"):
        for report in reports:
            await asyncio.sleep(1)
            try:
                ForecastDatabaseManager.add_forecast_report_to_database(
                    report, ForecastRunType.REGULAR_FORECAST
                )
            except Exception as e:
                logger.error(f"Error adding forecast report to database: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run forecasts with specified bot type"
    )
    parser.add_argument(
        "--bot-type",
        choices=["template", "main", "template_v1"],
        default="main",
        help="Type of bot to use for forecasting",
    )
    parser.add_argument(
        "--allow-rerun",
        action="store_true",
        help="Allow rerunning forecasts",
    )
    args = parser.parse_args()

    asyncio.run(run_morning_forecasts(args.bot_type, args.allow_rerun))
