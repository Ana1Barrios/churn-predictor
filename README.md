# Customer Churn Predictor

A data pipeline that collects repository activity data from the GitHub API, engineers features, and predicts customer (contributor) churn.

## Project Structure

- `scraper.py` – Collects and explores raw data from the GitHub API.
- `features.py` – Generates engineered features (ratio, time-based, aggregation, binary).
- `model.py` – Trains and evaluates the churn prediction model.
- `main.py` – Runs the full pipeline.
- `data/raw/` – Raw collected data.
- `notebooks/` – Exploratory analysis notebooks.

## Setup

1. Clone the repository:
   ```bash
   git clone <repo-url>
   cd <repo-name>
   ```
## Running with Docker

Build the container:
```bash
docker compose build
```

Run the container:
```bash
docker compose up
```

Run in the background (detached mode):
```bash
docker compose up -d
```

Stop the container:
```bash
docker compose down
```

Check container health:
```bash
docker compose ps
```



## License

MIT
