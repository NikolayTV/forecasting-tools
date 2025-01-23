import os
from datetime import datetime
from forecasting_tools.ai_models.ai_utils.ai_misc import clean_indents
from forecasting_tools.ai_models.deepseek_r1 import DeepSeekR1
from forecasting_tools.ai_models.gpto1preview import GptO1Preview
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
from forecasting_tools.ai_models.gpt4o import Gpt4o, Gpt4oTrue
logger = logging.getLogger(__name__)

class Q4VeritasWithExaAndPerplexity(Q4VeritasBot):
    FINAL_DECISION_LLM = Gpt4o(temperature=0)
    # FINAL_DECISION_LLM = DeepSeekR1(temperature=0.5)
    # FINAL_DECISION_LLM = GptO1Preview()

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
        use_compile_report: bool = False,
        num_searches_to_run: int = 1,
        num_quotes_to_evaluate_from_search: int = 10,
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
        self.use_compile_report = use_compile_report
        self.num_searches_to_run = num_searches_to_run
        self.num_quotes_to_evaluate_from_search = num_quotes_to_evaluate_from_search
    async def run_research(self, question: MetaculusQuestion) -> str:
        prompt = clean_indents(f"""
            You are an assistant to a superforecaster.
            The superforecaster will give you a question they intend to forecast on.
            To be a great assistant, you generate a concise but detailed rundown of the most relevant news, including if the question would resolve Yes or No based on current information.
            You do not produce forecasts yourself.""")

        response = ""
        if question.open_time and question.open_time.strftime("%Y-%m-%d") == datetime.today().strftime("%Y-%m-%d") and os.getenv("PERPLEXITY_API_KEY"):
            response += 'Online perplexity search conclusion:'
            perp_response = await Perplexity(system_prompt=prompt).invoke(question.question_text)
            response += str(perp_response)
            if not perp_response:
                logger.warning(f"PERPLEXITY FAILED on question: {question.question_text[:100]}")
        if os.getenv("EXA_API_KEY"):
            response += 'Internet search conclusion:'
            exa_response = await SmartSearcher(temperature=0.7,
                    num_searches_to_run=self.num_searches_to_run,
                    num_quotes_to_evaluate_from_search=self.num_quotes_to_evaluate_from_search,
                    use_compile_report=self.use_compile_report)\
                .invoke(question.question_text, end_published_date=question.open_time)
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
        prompt = clean_indents(f"""
            You are a professional forecaster participating in a job interview. Your task is to analyze a given question and provide a probability forecast based on the information provided. Please approach this task with the utmost professionalism and attention to detail.

            Here is the interview question you need to address:
            <question>
            {question.question_text}
            </question>

            The resolution criteria for this question are as follows:
            <resolution_criteria>
            {question.resolution_criteria if question.resolution_criteria else "No resolution criteria provided."}
            </resolution_criteria>

            Your internet search assistant has provided the following relevant information:
            <internet_search>
            {research}
            </internet_search>

            Before providing your final answer, please conduct a thorough Chain of Thought (COT) analysis. Wrap your analysis in <forecast_analysis> tags. Your analysis should cover the following points in detail:

            a) Key Information: Write down the most relevant quotes from the internet search and resolution criteria. Number each quote.

            b) Time Horizon: Carefully assess the time remaining until the outcome of the question. Consider how this timeframe might affect the certainty of your prediction.

            c) Initial Assessment: Provide an initial answer based on the information from the internet search and your common sense understanding. Explain your reasoning clearly.

            d) Influencing Factors: Identify and analyze the most crucial factors that could lead to either a successful or unsuccessful resolution. Consider both obvious and less apparent influences.

            e) Historical Precedent: Examine how often similar situations have occurred in the past. What does historical data suggest about the likelihood of this outcome? Be sure to explain how you're applying historical information to the current scenario.

            f) Conflicting Evidence: Identify any contradictions in the available information. How might these affect your forecast?

            g) Alternative Scenarios: Consider at least two alternative scenarios that could lead to different outcomes. How likely are these scenarios?

            h) Confidence Assessment: Reflect on your level of confidence in your analysis. What additional information would help improve your forecast?

            i) Summary: Summarize your confidence level and provide your final probability estimate as a percentage between 0 and 100.

            After completing your analysis, provide your final probability estimate as a percentage between 0 and 100.

            Your output should follow this structure:

            <analysis>
            [Your detailed analysis covering points a through i]
            </analysis>

            <final_answer>
            Probability: [Your final estimate]%
            </final_answer>

            Remember, your goal is to provide a well-reasoned, balanced forecast that takes into account all available information and potential scenarios. Your ability to think critically and provide clear, logical reasoning is crucial for this task.

            The current date is:
            <current_date>
            {question.open_time.strftime("%Y-%m-%d") if question.open_time else datetime.now().strftime("%Y-%m-%d")}
            </current_date>

        """)

        gpt_forecast = await self.FINAL_DECISION_LLM.invoke(prompt)
        final_answer = gpt_forecast.split("<final_answer>")[1].split("</final_answer>")[0]
        prediction = self._extract_forecast_from_binary_rationale(
            final_answer, max_prediction=0.95, min_prediction=0.05
        )

        return ReasonedPrediction(
            prediction_value=prediction, reasoning=gpt_forecast
        )

