"""Build a report and figures only from completed, verified experiment tables."""

from experiment.reporting import build_report, figures
from experiment.storage import begin_stage, finish_stage


def main():
    begin_stage("report", ("embed", "encode", "probe", "search", "subtract"))
    images = figures()
    report = build_report()
    finish_stage("report", [report, *images])
    print(f"Report: {report}")


if __name__ == "__main__":
    main()
