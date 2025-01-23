import os
from datetime import datetime
from forecasting_tools.ai_models.ai_utils.ai_misc import clean_indents
from forecasting_tools.ai_models.deepseek_r1 import DeepSeekR1
from forecasting_tools.forecasting.forecast_bots.official_bots.q4_veritas_bot import (
    Q4VeritasBot,
)
from forecasting_tools.forecasting.helpers.smart_searcher_nik import SmartSearcher
from forecasting_tools.forecasting.questions_and_reports.questions import (
    MetaculusQuestion,
)
from forecasting_tools.forecasting.questions_and_reports.forecast_report import (
    ReasonedPrediction,
)
from forecasting_tools.forecasting.questions_and_reports.questions import (
    BinaryQuestion,
    MetaculusQuestion,
)
from forecasting_tools.ai_models.perplexity import Perplexity
import logging

from forecasting_tools.forecasting.sub_question_researchers.research_coordinator import ResearchCoordinator
logger = logging.getLogger(__name__)
class Q4VeritasWithExa(Q4VeritasBot):
    FINAL_DECISION_LLM = DeepSeekR1(temperature=0.5)

    def __init__(
        self,
        research_reports_per_question: int = 1,
        predictions_per_research_report: int = 1,
        use_research_summary_to_forecast: bool = False,
        publish_reports_to_metaculus: bool = False,
        folder_to_save_reports_to: str | None = None,
        skip_previously_forecasted_questions: bool = False,
        number_of_background_questions_to_ask: int = 0,
        number_of_base_rate_questions_to_ask: int = 0,
        number_of_base_rates_to_do_deep_research_on: int = 0,
        **kwargs,
    ) -> None:
        super().__init__(
            research_reports_per_question=research_reports_per_question,
            predictions_per_research_report=predictions_per_research_report,
            use_research_summary_to_forecast=use_research_summary_to_forecast,
            publish_reports_to_metaculus=publish_reports_to_metaculus,
            folder_to_save_reports_to=folder_to_save_reports_to,
            skip_previously_forecasted_questions=skip_previously_forecasted_questions,
            number_of_background_questions_to_ask=number_of_background_questions_to_ask,
            number_of_base_rate_questions_to_ask=number_of_base_rate_questions_to_ask,
            number_of_base_rates_to_do_deep_research_on=number_of_base_rates_to_do_deep_research_on,
            **kwargs,
        )

    async def run_research(self, question: MetaculusQuestion) -> str:
        prompt = clean_indents(
            f"""
            You are an assistant to a superforecaster.
            The superforecaster will give you a question they intend to forecast on.
            To be a great assistant, you generate a concise but detailed rundown of the most relevant news, including if the question would resolve Yes or No based on current information.
            You do not produce forecasts yourself.

            Question:
            {question.question_text}
            """
        )

        response = ""
        if question.open_time and question.open_time.strftime("%Y-%m-%d") == datetime.today().strftime("%Y-%m-%d") and os.getenv("PERPLEXITY_API_KEY"):
            response += 'Online perplexity search conclusion:'
            perp_response = await Perplexity(system_prompt=prompt).invoke(prompt)
            response += str(perp_response)
            if not perp_response:
                logger.warning(f"PERPLEXITY FAILED on question: {question.question_text[:100]}")
        if os.getenv("EXA_API_KEY"):
            response += 'Internet search conclusion:'
            exa_response = await SmartSearcher(temperature=0.4).invoke(prompt, end_published_date=question.open_time)
            response += str(exa_response)
            if not exa_response:
                logger.warning(f"EXA FAILED on question: {question.question_text[:100]}")
        else:
            logger.error(
                "No API keys for searching the web. Skipping research and setting it blank."
            )
            response = ""
        return response

    async def summarize_research(
        self, question: MetaculusQuestion, research: str
    ) -> str:
        # research_coordinator = ResearchCoordinator(question)
        # summary_report = (
        #     await research_coordinator.summarize_full_research_report(research)
        # )
        return research

    async def _run_forecast_on_binary(
        self, question: BinaryQuestion, research: str
    ) -> ReasonedPrediction[float]:
        assert isinstance(
            question, BinaryQuestion
        ), "Question must be a BinaryQuestion"
        prompt = clean_indents(
            f"""
            You are a professional forecaster interviewing for a job.
            Your interview question is:
            <question>
            {question.question_text}
            </question>

            Resolution criteria:
            <resolution_criteria>
            {question.resolution_criteria if question.resolution_criteria else "No resolution criteria provided."}
            </resolution_criteria>

            Your internet search assistant says:
            <internet_search>
            {research}
            </internet_search>

            Today is {question.open_time.strftime("%Y-%m-%d") if question.open_time else datetime.now().strftime("%Y-%m-%d")}.


            Before answering you do COT analysis in <analysis> tags, then provide the JSON output.

            (a) The time left until the outcome to the question is known
            (b) The most obvious answer to the question provided with information from the internet search and common sense.
            (c) The most important factors that will influence a successful/unsuccessful resolution.
            (d) What you would forecast if you were to only use historical precedent (i.e. how often this happens in the past) without any current information.
            (e) The last thing you write is your final answer as: "Probability: ZZ%", 0-100

            Example output:
            <analysis>
            ...
            </analysis>
            <final_answer>
            Probability: ZZ%
            </final_answer>
            """
        )
        gpt_forecast = await self.FINAL_DECISION_LLM.invoke(prompt)
        final_answer = gpt_forecast.split("<final_answer>")[1].split("</final_answer>")[0]
        prediction = self._extract_forecast_from_binary_rationale(
            final_answer, max_prediction=0.95, min_prediction=0.05
        )
        reasoning = (
            gpt_forecast
            + "\nThe original forecast may have been clamped between 5% and 95%."
        )
        return ReasonedPrediction(
            prediction_value=prediction, reasoning=reasoning
        )

