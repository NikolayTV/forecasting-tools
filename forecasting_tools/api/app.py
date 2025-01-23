from datetime import datetime
import logging
import os
import asyncio
from fastapi import FastAPI, HTTPException, Depends, Security
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel
import uvicorn
from forecasting_tools.forecasting.forecast_bots.official_bots.q4_veritas_bot import Q4VeritasBot
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

@app.get("/get_prediction")
async def get_prediction(
    question: str,
    current_date: str | None = None,
    version: str | None = None,
    authorized: bool = Depends(verify_token)
) -> PredictionResponse:
    try:
        async with request_semaphore:  # This will wait if there are already 20 active requests
            if current_date:
                try:
                    date = datetime.strptime(current_date, "%Y-%m-%d")
                except ValueError:
                    raise HTTPException(status_code=400, detail="Date must be in YYYY-MM-DD format")
            else:
                date = datetime.now()
            if version == "v2":
                bot = TemplateBot_v1(
                    research_reports_per_question=3,
                    predictions_per_research_report=3,
                    use_research_summary_to_forecast=True,
                    publish_reports_to_metaculus=False,
                    folder_to_save_reports_to='./reports',
                    skip_previously_forecasted_questions=True,
                )
            elif version == "Q1":
                report = await Q4VeritasBot(
                    research_reports_per_question=3,
                    predictions_per_research_report=3,
                    publish_reports_to_metaculus=False,
                    folder_to_save_reports_to='./reports',
                    number_of_background_questions_to_ask=5,
                    number_of_base_rate_questions_to_ask=5,
                    number_of_base_rates_to_do_deep_research_on=0,
                ).forecast_question(input.question)
            else:
                bot = TemplateBot_v1(
                    research_reports_per_question=1,
                    predictions_per_research_report=1,
                    use_research_summary_to_forecast=True,
                    publish_reports_to_metaculus=False,
                    folder_to_save_reports_to='./reports',
                skip_previously_forecasted_questions=True,
            )

            # Create question object
            question_obj = BinaryQuestion(
                question_text=question,
                background_info="",
                resolution_criteria="",
                fine_print="",
                state=QuestionState.OPEN,
                open_time=date,
                id_of_post=0,
            )

            # Get prediction
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
        logger.error(f"Error in get_prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(
        "forecasting_tools.api.app:app",
        host="0.0.0.0",
        port=8000,
        log_level="info",
        reload=False,
        loop="asyncio",
    )