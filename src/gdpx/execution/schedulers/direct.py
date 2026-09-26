"""Synchronous execution without a queue manager."""

import subprocess

from .scheduler import BaseScheduler


class DirectScheduler(BaseScheduler):
    """Run work synchronously on the host selected by the transport."""

    name = "direct"
    is_direct = True
    supports_concurrent_tasks = True
    SHELL = "#!/bin/bash -l"

    @BaseScheduler.job_name.setter
    def job_name(self, value: str) -> None:
        self._job_name = value

    def is_finished(self) -> bool:
        return True

    def submit(self, func_to_execute=None) -> str:
        if self.is_dry_run:
            return f"Attempt to run {self.script.name} directly."
        if func_to_execute is not None:
            func_to_execute()
        else:
            if not self.script.exists():
                self.write()
            subprocess.run(["bash", "-l", self.script.name], cwd=self.script.parent, check=True)
        return "direct"

    def __str__(self) -> str:
        content = self.SHELL + self._convert_environs_to_content()
        if self.user_commands:
            content += "\n" + self.user_commands
        return content


__all__ = ["DirectScheduler"]
