# Forecasting API

A FastAPI application that provides predictions using the TemplateBot_v1 forecasting model.

## Setup

### Option 1: Local Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up environment variables:
- `OPENAI_API_KEY` - OpenAI API key
- `METACULUS_TOKEN` - Metaculus API token (optional)
- `ANTHROPIC_API_KEY` - Anthropic API key (optional)
- `PERPLEXITY_API_KEY` - Perplexity API key (optional)
- `EXA_API_KEY` - Exa API key (optional)

3. Start the server:
```bash
uvicorn forecasting_tools.api.app:app --reload
```

### Option 2: Docker Installation

1. Make sure you have Docker and Docker Compose installed

2. Create a `.env` file with your environment variables:
```bash
OPENAI_API_KEY=your_key_here
METACULUS_TOKEN=your_token_here  # Optional
ANTHROPIC_API_KEY=your_key_here  # Optional
PERPLEXITY_API_KEY=your_key_here # Optional
EXA_API_KEY=your_key_here        # Optional
```

3. Create a `reports` directory for saving prediction reports:
```bash
mkdir reports
```

4. Build and start the container:
```bash
docker-compose up --build
```

The API will be available at http://localhost:8000 and reports will be saved to your local `reports` directory.

## API Endpoints

### GET /get_prediction

Get a prediction for a binary question.

**Parameters:**
- `question` (string): The question text followed by resolution criteria, separated by a question mark
- `current_date` (string): The date in YYYY-MM-DD format
- `version` (string): Version identifier (currently not used)

**Example Request:**
```bash
curl "http://localhost:8000/get_prediction?question=Will%20SpaceX%20launch%20Starship%20in%202024?%20Resolution%20criteria:%20SpaceX%20must%20successfully%20launch%20Starship%20to%20orbit&current_date=2024-01-01&version=v1"
```

**Example Response:**
```json
{
    "prediction": 0.75,
    "reasoning": "Based on current progress and historical data...",
    "minutes_taken": 2.5,
    "price_estimate": 0.15
}
```

**Error Responses:**
- 400: Bad Request (invalid question format or date)
- 500: Internal Server Error