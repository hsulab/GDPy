"""List active simulations independently of any working directory."""

import json

from gdpx.execution.processes import running_jobs


def run_queue(*, all_users=False, as_json=False, long=False):
    jobs = running_jobs(all_users=all_users)
    if as_json:
        print(json.dumps(jobs))
        return
    columns = ("JOBID", "PID", "USER", "STAT", "ELAPSED", "DIRECTORY")
    keys = ("job_id", "pid", "user", "state", "elapsed", "directory")
    if long:
        columns += ("COMMAND",)
        keys += ("command",)
    rows = [[str(job[key]) if job[key] is not None else "—" for key in keys] for job in jobs]
    if not long:
        for row in rows:
            if row[0].startswith("nohup-"):
                row[0] = row[0][:14]  # Provider prefix and the first UUID segment.
            if len(row[5]) > 32:
                row[5] = "…" + row[5][-31:]
    widths = [max(len(column), *(len(row[index]) for row in rows)) if rows else len(column)
              for index, column in enumerate(columns)]
    for row in [columns, *rows]:
        print("  ".join(value.ljust(width) for value, width in zip(row, widths)).rstrip())
