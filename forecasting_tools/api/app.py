from datetime import datetime
import logging
import os
import asyncio
from fastapi import FastAPI, HTTPException, Depends, Security, Query
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel
import uvicorn
from telegram import Bot
from forecasting_tools.forecasting.forecast_bots.experiments.q4v_w_exa_nik import Q4VeritasWithExaAndPerplexity
from forecasting_tools.forecasting.forecast_bots.template_v1_bot import TemplateBot_v1
from forecasting_tools.forecasting.questions_and_reports.questions import BinaryQuestion, QuestionState

logger = logging.getLogger(__name__)

app = FastAPI()
security = HTTPBearer()
request_semaphore = asyncio.Semaphore(10)  # Limit to 10 concurrent requests

env_vars = [
    "OPENAI_API_KEY",
    "METACULUS_TOKEN",
    "ANTHROPIC_API_KEY",
    "PERPLEXITY_API_KEY",
    "EXA_API_KEY",
    "CODA_API_KEY",
    "HUGGINGFACE_API_KEY",
    "API_TOKEN"
]

def verify_token(credentials: HTTPAuthorizationCredentials = Security(security)) -> bool:
    token = os.getenv("API_TOKEN")
    if not token:
        raise HTTPException(status_code=500, detail="API token not configured")
    if credentials.credentials != token:
        raise HTTPException(status_code=401, detail="Invalid token")
    return True

def split_question_criteria(text: str) -> tuple[str, str]:
    """
    Splits text into question and resolution criteria.

    Args:
        text: Input text containing question and criteria

    Returns:
        Tuple of (question, criteria)
    """
    # Split on first question mark
    parts = text.split('?', 1)

    if len(parts) != 2:
        return text, ""

    question = parts[0] + '?'  # Add back the question mark
    criteria = parts[1].strip()  # Remove leading/trailing whitespace

    return question, criteria

if __name__ == "__main__":
    print("\nEnvironment variables status:", flush=True)
    for var in env_vars:
        value = os.getenv(var)
        is_set = "✓" if value else "✗"
        # Only show first/last 4 chars if value exists
        display_value = f"{value[:4]}...{value[-4:]}" if value else "Not set"
        print(f"{var}: {is_set} {display_value}", flush=True)


class PredictionResponse(BaseModel):
    prediction: float
    reasoning: str
    minutes_taken: float | None
    price_estimate: float | None

class TelegramNotifier:
    def __init__(self):
        token = os.getenv("TELEGRAM_BOT_TOKEN")
        chat_id = os.getenv("TELEGRAM_CHAT_ID")
        warning_thread = os.getenv("TELEGRAM_WARNING_THREAD_ID")
        error_thread = os.getenv("TELEGRAM_ERROR_THREAD_ID")

        if not all([token, chat_id, warning_thread, error_thread]):
            raise ValueError("Missing required Telegram environment variables")

        self.bot = Bot(token=str(token))
        self.chat_id = str(chat_id)
        self.warning_thread_id = int(str(warning_thread))
        self.error_thread_id = int(str(error_thread))

    async def send_warning(self, msg: str):
        resp = await self.bot.send_message(
            chat_id=self.chat_id,
            text=msg[:4096],
            parse_mode=None,
            message_thread_id=self.warning_thread_id
        )
        print(resp)

    async def send_error(self, msg: str):
        resp = await self.bot.send_message(
            chat_id=self.chat_id,
            text=msg[:4096],
            parse_mode=None,
            message_thread_id=int(self.error_thread_id)
        )
        print(resp)

@app.get("/get_prediction")
async def get_prediction(
    question: str = Query(..., description="The question to predict"),
    current_date: str | None = Query(None, description="Optional date in YYYY-MM-DD format. Used as cutoff for news. Defaults to current date"),
    version: str | None = Query("v1", description="Model version to use - 'v1' (faster ~40s, basic) or 'v2' (better but slower ~70s, more thorough)"),
    authorized: bool = Depends(verify_token)
) -> PredictionResponse:
    """Get a prediction for a given question.

    Args:
        question: The question to predict
        current_date: Optional date in YYYY-MM-DD format. Defaults to current date
        version: Model version to use - 'v1' (faster ~40s, basic) or 'v2' (better but slower ~70s, more thorough)
        authorized: Automatically injected auth status
    """
    try:
        notifier = TelegramNotifier()
    except ValueError as e:
        logger.warning(f"Telegram notifications disabled: {e}")
        notifier = None

    max_retries = 5
    retry_count = 0
    last_error = None

    while retry_count < max_retries:
        try:
            async with request_semaphore:
                if current_date:
                    try:
                        date = datetime.strptime(current_date, "%Y-%m-%d")
                    except ValueError:
                        raise HTTPException(status_code=400, detail="Date must be in YYYY-MM-DD format")
                else:
                    date = datetime.now()
                if version == 'v1':
                    bot = TemplateBot_v1(
                        research_reports_per_question=1,
                        predictions_per_research_report=1,
                        use_research_summary_to_forecast=True,
                        publish_reports_to_metaculus=False,
                        folder_to_save_reports_to='./reports',
                        skip_previously_forecasted_questions=True,
                    )
                elif version == "v2":
                    bot = Q4VeritasWithExaAndPerplexity(
                        research_reports_per_question=2,
                        predictions_per_research_report=3,
                        num_searches_to_run=3,
                        num_quotes_to_evaluate_from_search=10,
                        publish_reports_to_metaculus=False,
                        folder_to_save_reports_to='./reports',
                    )
                else:
                    raise ValueError("Invalid version")

                question_text, criteria = split_question_criteria(question)
                question_obj = BinaryQuestion(
                    question_text=question_text,
                    background_info="",
                    resolution_criteria=criteria,
                    fine_print="",
                    state=QuestionState.OPEN,
                    open_time=date,
                    id_of_post=0,
                )

                reports = await bot.forecast_questions([question_obj])

                if not reports:
                    raise HTTPException(status_code=500, detail="No prediction returned")

                report = reports[0]
                report_json = report.to_json()

                return PredictionResponse(
                    prediction=report.prediction,
                    reasoning=report.explanation,
                    minutes_taken=report_json.get('minutes_taken'),
                    price_estimate=report_json.get('price_estimate')
                )

        except Exception as e:
            retry_count += 1
            last_error = e
            error_msg = f"Error in get_prediction (attempt {retry_count}/{max_retries}): {str(e)}"
            logger.error(error_msg)

            if retry_count < max_retries:
                if notifier:
                    await notifier.send_warning(error_msg)
                await asyncio.sleep(2 ** retry_count)  # Exponential backoff
            else:
                if notifier:
                    await notifier.send_error(f"Final error in get_prediction after {max_retries} attempts: {str(e)}")
                raise HTTPException(status_code=500, detail=str(last_error))

    # This should never be reached due to the raise in the else block above
    raise HTTPException(status_code=500, detail="Unexpected error in retry loop")

if __name__ == "__main__":
    uvicorn.run(
        "forecasting_tools.api.app:app",
        host="0.0.0.0",
        port=8000,
        log_level="info",
        reload=False,
        loop="asyncio",
    )