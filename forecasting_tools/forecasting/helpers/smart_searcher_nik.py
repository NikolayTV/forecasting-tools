import asyncio
import logging
import re
import urllib.parse
from datetime import datetime
from json_repair import repair_json
import json

from forecasting_tools.ai_models.ai_utils.ai_misc import clean_indents
from forecasting_tools.ai_models.basic_model_interfaces.ai_model import AiModel
from forecasting_tools.ai_models.basic_model_interfaces.outputs_text import (
    OutputsText,
)
from forecasting_tools.ai_models.exa_searcher import (
    ExaHighlightQuote,
    ExaSearcher,
    SearchInput,
)
from forecasting_tools.forecasting.helpers.configured_llms import BasicLlm
from forecasting_tools.forecasting.helpers.works_cited_creator import (
    WorksCitedCreator,
)

logger = logging.getLogger(__name__)


class SmartSearcher(OutputsText, AiModel):
    """
    Answers a prompt, using search results to inform its response.
    """

    def __init__(
        self,
        *args,
        temperature: float = 0,
        include_works_cited_list: bool = False,
        use_brackets_around_citations: bool = True,
        num_searches_to_run: int = 2,
        num_sites_per_search: int = 10,
        use_compile_report: bool = False,
        num_quotes_to_evaluate_from_search: int = 10,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        assert 0 <= temperature <= 1, "Temperature must be between 0 and 1"
        self.temperature = temperature
        self.num_quotes_to_evaluate_from_search = num_quotes_to_evaluate_from_search
        self.number_of_searches_to_run = num_searches_to_run
        self.exa_searcher = ExaSearcher(
            include_text=False,
            include_highlights=True,
            num_results=num_sites_per_search,
        )
        self.llm = BasicLlm(temperature=temperature)
        self.include_works_cited_list = include_works_cited_list
        self.use_citation_brackets = use_brackets_around_citations
        self.use_compile_report = use_compile_report

    async def invoke(self, prompt: str, end_published_date: datetime | None = None) -> str:
        logger.debug(f"Running search for prompt: {prompt}")
        if not prompt:
            raise ValueError("Prompt cannot be empty")
        report, _ = await self._mockable_direct_call_to_model(prompt, end_published_date=end_published_date)
        logger.debug(f"Report: {report[:1000]}...")
        return report

    async def _mockable_direct_call_to_model(
        self, prompt: str, end_published_date: datetime | None = None
    ) -> tuple[str, list[ExaHighlightQuote]]:
        search_terms = await self.__come_up_with_search_queries(prompt, end_published_date=end_published_date)
        quotes = await self.__search_for_quotes(search_terms)
        if self.use_compile_report:
            report = await self.__compile_report(quotes, prompt, end_published_date=end_published_date) # TODO - возможно лишний шаг, можно напрямую закинуть цитаты в промт
            if self.include_works_cited_list:
                works_cited_list = WorksCitedCreator.create_works_cited_list(
                    quotes, report
                )
                report = report + "\n\n" + works_cited_list
                final_report = self.__add_links_to_citations(report, quotes)
            else:
                final_report = report
        else:
            highlights_context = self._turn_highlights_into_search_context_for_prompt(
                quotes
            )
            final_report = str(highlights_context)
        return final_report, quotes

    @staticmethod
    def __extract_search_inputs_json(response: str) -> str:
        pattern = r"<search_inputs>\s*(\[[\s\S]*?\])\s*</search_inputs>"
        match = re.search(pattern, response)
        if not match:
            print("Search query generation parsing failed:")
            print(response)
            raise ValueError("No search inputs JSON found in response")
        return match.group(1)

    async def __come_up_with_search_queries(
        self, prompt: str, end_published_date: datetime | None = None
    ) -> list[SearchInput]:
        prompt = clean_indents(
            f"""
            TODAY'S DATE IS {end_published_date.strftime('%Y-%m-%d') if end_published_date else datetime.now().strftime('%Y-%m-%d')}

            You are an assistant to a superforecaster.
            The superforecaster will give you a question they intend to forecast on.
            To be a great assistant, you generate a concise but detailed rundown of the most relevant news, including if the question would resolve Yes or No based on current information.
            You do not produce forecasts yourself.

            You have been given the following question to analyze:
            <question>
            {prompt}
            </question>

            Generate {self.number_of_searches_to_run} google searches that will be usefull you to answer the question.
            Each query should be unique and not be a variation of another query.

            First, provide your COT analysis in <analysis> tags, then provide the JSON output.
            Example questions to consider:
            - Take into account current date when generating start_published_date and relevant queries
            - Note that the most recent news are the most relevant, therefore make reasanable limit on start_published_date for the news that are very abundant
            - What are the aspects of the question that are most important? Are there multiple aspects?

            Return a list of search inputs as a list of JSON objects. Each object must have this schema:
            Here is the search_inputs schema as a list of JSON objects:
            [
                {{
                    "web_search_query": "Required. The search query to find relevant web pages",
                    "highlight_query": "Optional. Text to highlight within found documents. If not provided, web_search_query will be used",
                    "start_published_date": "Optional. Earliest allowed publication date in ISO format. Default: null"
                }}
            ]

            Example output:
            <analysis>
            ...
            </analysis>
            <search_inputs>
            [
                {{
                    "web_search_query": "...",
                    "highlight_query": "...",
                    "start_published_date": "..."
                }},
                {{
                    "web_search_query": "...",
                    "highlight_query": "...",
                    "start_published_date": "..."
                }}
            ]
            </search_inputs>
            """
        )
        max_retries = 3
        retry_delay = 2  # seconds
        last_error = None

        for attempt in range(max_retries):
            try:
                response = await self.llm.invoke(prompt)
                json_str = self.__extract_search_inputs_json(response)
                repaired_json = json.loads(str(repair_json(json_str)))
                break
            except Exception as e:
                last_error = e
                if attempt < max_retries - 1:
                    logger.warning(f"Attempt {attempt + 1} failed with error: {str(e)}. Retrying in {retry_delay} seconds...")
                    await asyncio.sleep(retry_delay * (attempt + 1))
                continue
        else:
            error_msg = f"Failed to generate search inputs after {max_retries} attempts. Last error: {str(last_error)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)

        search_terms = [SearchInput(**search) for search in repaired_json]
        for search in search_terms:
            search.end_published_date = end_published_date
        search_log = "\n".join(
            [
                f"Search {i+1}: {search}"
                for i, search in enumerate(search_terms)
            ]
        )
        logger.info(f"Decided on searches:\n{search_log}")
        return search_terms

    async def __search_for_quotes(
        self, search_inputs: list[SearchInput]
    ) -> list[ExaHighlightQuote]:
        async def try_search() -> list[list[ExaHighlightQuote]]:
            return await asyncio.gather(
                *[
                    self.exa_searcher.invoke_for_highlights_in_relevance_order(
                        search_query_or_strategy=search
                    )
                    for search in search_inputs
                ]
            )

        max_retries = 7
        attempt = 1

        while attempt <= max_retries:
            logger.info(f"Search attempt {attempt}/{max_retries}")
            all_quotes = await try_search()
            flattened_quotes = [
                quote for sublist in all_quotes for quote in sublist
            ]
            unique_quotes: dict[str, ExaHighlightQuote] = {}
            for quote in flattened_quotes:
                if quote.highlight_text not in unique_quotes:
                    unique_quotes[quote.highlight_text] = quote
            deduplicated_quotes = sorted(
                unique_quotes.values(), key=lambda x: x.score, reverse=True
            )

            if len(deduplicated_quotes) > 0:
                break

            if attempt < max_retries:
                logger.warning(f"No quotes found on attempt {attempt}, retrying search...")
                await asyncio.sleep(2 * attempt)
            attempt += 1

        if len(deduplicated_quotes) == 0:
            raise RuntimeError(f"No quotes found after {max_retries} attempts")

        if len(deduplicated_quotes) < self.num_quotes_to_evaluate_from_search:
            logger.warning(
                f"Couldn't find the number of quotes asked for. Found {len(deduplicated_quotes)} quotes, but need {self.num_quotes_to_evaluate_from_search} quotes"
            )
        most_relevant_quotes = deduplicated_quotes[
            : self.num_quotes_to_evaluate_from_search
        ]
        return most_relevant_quotes

    async def __compile_report(
        self,
        search_results: list[ExaHighlightQuote],
        original_instructions: str,
        end_published_date: datetime | None = None,
    ) -> str:
        if len(search_results) == 0:
            return "No search results found for the query using the search filter chosen"

        assert (
            len(search_results) <= self.num_quotes_to_evaluate_from_search
        ), "Too many search results found"

        search_result_context = (
            self._turn_highlights_into_search_context_for_prompt(
                search_results
            )
        )
        logger.info(f"Generating response using {len(search_results)} quotes")
        logger.debug(f"Search results:\n{search_result_context}")
        prompt = clean_indents(
            f"""
            Today is {end_published_date.strftime("%Y-%m-%d") if end_published_date else datetime.now().strftime("%Y-%m-%d")}.
            You have been given the following qustion

            <question>
            {original_instructions}
            </question>

            After searching the internet, you found the following results.
            <internet_search_results>
            {search_result_context}
            </internet_search_results>

            Please answer the question using the search results.
            Consider relative dates of the search results and the today's date when providing your answer.

            Please cite your sources inline and use markdown formatting.

            Clearly state:
             - is your response based on the provided sources or not
             - how confident you are in the provided sources.
             - date of publication of the used source

            For instance, this quote:
            > [1] "SpaceX successfully completed a full flight test of its Starship spacecraft on April 20, 2023"

            Would be cited like this:
            > SpaceX successfully completed a full flight test of its Starship spacecraft on April 20, 2023 [1].
            """
        )
        report = await self.llm.invoke(prompt)
        return report

    @staticmethod
    def _turn_highlights_into_search_context_for_prompt(
        highlights: list[ExaHighlightQuote],
    ) -> str:
        search_context = ""
        for i, highlight in enumerate(highlights):
            url = highlight.source.url
            title = highlight.source.title
            publish_date = highlight.source.readable_publish_date
            search_context += f'[{i+1}] "{highlight.highlight_text}". [This quote is from {url} titled "{title}", published on {publish_date}]\n'
        return search_context

    def __add_links_to_citations(
        self, report: str, highlights: list[ExaHighlightQuote]
    ) -> str:
        for i, highlight in enumerate(highlights):
            citation_num = i + 1
            less_than_10_words = len(highlight.highlight_text.split()) < 10
            if less_than_10_words:
                text_fragment = highlight.highlight_text
            else:
                first_five_words = " ".join(
                    highlight.highlight_text.split()[:5]
                )
                last_five_words = " ".join(
                    highlight.highlight_text.split()[-5:]
                )
                encoded_first_five_words = urllib.parse.quote(
                    first_five_words, safe=""
                )
                encoded_last_five_words = urllib.parse.quote(
                    last_five_words, safe=""
                )
                text_fragment = f"{encoded_first_five_words},{encoded_last_five_words}"  # Comma indicates that anything can be included in between
            text_fragment = text_fragment.replace("(", "%28").replace(
                ")", "%29"
            )
            text_fragment = text_fragment.replace("-", "%2D").strip(",")
            text_fragment = text_fragment.replace(" ", "%20")
            fragment_url = f"{highlight.source.url}#:~:text={text_fragment}"

            if self.use_citation_brackets:
                markdown_url = f"\\[[{citation_num}]({fragment_url})\\]"
            else:
                markdown_url = f"[{citation_num}]({fragment_url})"

            # Combined regex pattern for all citation types
            pattern = re.compile(
                r"(?:\\\[)?(\[{}\](?:\(.*?\))?)(?:\\\])?".format(citation_num)
            )
            # Matches:
            # [1]
            # [1](some text)
            # \[[1]\]
            # \[[1](some text)\]
            report = pattern.sub(markdown_url, report)

        return report

    @staticmethod
    def _get_cheap_input_for_invoke() -> str:
        return "What is the recent news on SpaceX?"

    @staticmethod
    def _get_mock_return_for_direct_call_to_model_using_cheap_input() -> str:
        return "Mock Report: Pretend this is an extensive report"
