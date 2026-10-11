import sys

import pytest

from gdpx.execution.schedulers.scheduler import JobSubmissionError, submit_job_script


def test_sbatch_rejection_preserves_reason_and_limit_hint(tmp_path):
    command = tmp_path / 'sbatch'
    command.write_text('#!/bin/sh\necho "sbatch: error: QOSMaxSubmitJobPerUserLimit" >&2\n'
                       'echo "sbatch: error: Batch job submission failed: Job violates accounting/QOS policy" >&2\nexit 1\n')
    command.chmod(0o755)
    script = tmp_path / 'job with spaces.slurm'
    script.write_text('#!/bin/sh\n')
    with pytest.raises(JobSubmissionError) as caught:
        submit_job_script(script, str(command), 5)
    assert caught.value.returncode == 1
    assert 'sbatch: error:' in caught.value.reason
    assert 'QOSMaxSubmitJobPerUserLimit' in str(caught.value)
    assert 'running/pending job limit' in str(caught.value)
    assert 'Job violates accounting/QOS policy' in str(caught.value)
    assert caught.value.summary == 'Job submission paused: Slurm job limit reached. Rerun when a slot is available.'
    caught.value.set_remaining_jobs(3, tmp_path / '0005.run_vasp')
    assert caught.value.summary == ('Job submission paused: Slurm job limit reached. '
                                  '3 jobs remain unsubmitted in 0005.run_vasp. Rerun when a slot is available.')
    caught.value.set_remaining_jobs(1, tmp_path / 'active')
    assert '1 job remains unsubmitted in active.' in caught.value.summary
    assert '3 jobs' not in caught.value.summary


def test_unknown_rejection_does_not_claim_submission_limit(tmp_path):
    command = tmp_path / 'sbatch'
    command.write_text('#!/bin/sh\necho "sbatch: error: Invalid account" >&2\nexit 1\n')
    command.chmod(0o755)
    with pytest.raises(JobSubmissionError, match='Invalid account') as caught:
        submit_job_script(tmp_path / 'job.slurm', str(command), 5)
    assert 'job limit' not in str(caught.value)
    assert caught.value.summary == 'Job submission failed: Invalid account'


def test_successful_submission_and_empty_response(tmp_path):
    command = tmp_path / 'sbatch'
    command.write_text('#!/bin/sh\nprintf "Submitted batch job 123\\n"\n')
    command.chmod(0o755)
    assert submit_job_script(tmp_path / 'job.slurm', str(command), 5) == '123'
    command.write_text('#!/bin/sh\nexit 0\n')
    with pytest.raises(JobSubmissionError, match='no job id'):
        submit_job_script(tmp_path / 'job.slurm', str(command), 5)


@pytest.mark.parametrize('subcommand', ['workflow', 'compute', 'train'])
def test_cli_reports_rejection_without_traceback(monkeypatch, tmp_path, subcommand):
    import gdpx.main as cli

    monkeypatch.setattr(sys, 'argv', ['gdp', '-d', str(tmp_path), subcommand,
                                    *(['run'] if subcommand == 'workflow' else []), 'input.yaml'])
    monkeypatch.setattr(cli, 'bootstrap_registries', lambda **kwargs: None)
    monkeypatch.setattr(cli, 'parse_input_file', lambda path: {})
    messages = []
    monkeypatch.setattr(cli.config.logger, 'info', messages.append)
    monkeypatch.setattr(cli.config.logger, 'error', lambda message: pytest.fail(message))
    monkeypatch.setattr(cli.config.logger, 'debug', lambda message: None)

    def reject(*args, **kwargs):
        error = JobSubmissionError(tmp_path / 'job.slurm', 'sbatch: error: QOSMaxSubmitJobPerUserLimit', 1)
        error.set_remaining_jobs(2, tmp_path / '0005.run_vasp')
        raise error

    if subcommand == 'workflow':
        monkeypatch.setattr('gdpx.cli.workflow.run_workflow', reject)
    elif subcommand == 'compute':
        monkeypatch.setattr('gdpx.cli.compute.run_computation', reject)
    else:
        monkeypatch.setattr('gdpx.cli.train.run_trainer', reject)
    assert cli.main() == 1
    assert len(messages) == 1
    assert messages[0] == ('Job submission paused: Slurm job limit reached. '
                           '2 jobs remain unsubmitted in 0005.run_vasp. Rerun when a slot is available.')
    assert 'Traceback' not in messages[0]


def test_cli_does_not_hide_unrelated_errors(monkeypatch):
    import gdpx.main as cli

    def broken():
        raise ValueError('bad configuration')

    monkeypatch.setattr(cli, '_main', broken)
    with pytest.raises(ValueError, match='bad configuration'):
        cli.main()
