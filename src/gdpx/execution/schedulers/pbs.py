#!/usr/bin/env python3
# -*- coding: utf-8 -*


from .scheduler import BaseScheduler


class PbsScheduler(BaseScheduler):

    name = "pbs"

    PREFIX = "#"
    SUFFIX = ".pbs"
    SHELL = "#!/bin/bash -l"

    SUBMIT_COMMAND = "qsub"
    ENQUIRE_COMMAND = "qstat"

    default_parameters = {}

    running_status = ["R", "Q", "PD", "CG"]

    def __str__(self):
        """Return the content of the job script."""
        content = self.SHELL + "\n"
        for key, value in self.parameters.items():
            if value:
                content += "{} --{}={}\n".format(self.PREFIX, key, value)
            # else:
            #    raise ValueError("Keyword *%s* not properly set." %key)

        content += self._convert_environs_to_content()

        if self.user_commands:
            content += "\n\n"
            content += self.user_commands

        return content

    def is_finished(self) -> bool:
        """Not implemented yet."""

        raise NotImplementedError()


if __name__ == "__main__":
    ...
