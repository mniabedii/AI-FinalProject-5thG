import json
from pathlib import Path
from datetime import datetime

def main():
    metrics_path = Path("reports/metrics/ci_metrics.json")
    metrics = json.loads(metrics_path.read_text())

    out = Path("reports/ci_report.md")
    out.write_text(
        f"# CI Report\n\n"
        f"- Generated: {datetime.now()}\n"
        f"- loss: {metrics['loss']}\n"
        f"- accuracy: {metrics['accuracy']}\n"
    )
    print("REPORT OK -> reports/ci_report.md")

if __name__ == "__main__":
    main()
