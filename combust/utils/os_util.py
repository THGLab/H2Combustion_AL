import subprocess
import sys
import os
from pathlib import Path
import contextlib
from typing import List, Union, Optional, Tuple


class CommandExecuteError(Exception):
    """
    Exception for command line exec error
    """

    def __init__(self, msg: str):
        self.msg = msg

    def __str__(self):
        return self.msg

    def __repr__(self):
        return self.msg

@contextlib.contextmanager
def set_directory(dirname: os.PathLike, mkdir: bool = False):
    """
    Set current workding directory within context

    Parameters
    ----------
    dirname : os.PathLike
        The directory path to change to
    mkdir: bool
        Whether make directory if `dirname` does not exist

    Yields
    ------
    path: Path
        The absolute path of the changed working directory

    Examples
    --------
    >>> with set_directory("some_path"):
    ...    do_something()
    """
    pwd = os.getcwd()
    path = Path(dirname).resolve()
    if mkdir:
        path.mkdir(exist_ok=True, parents=True)
    os.chdir(path)
    yield path
    os.chdir(pwd)


def run_command(
        cmd: Union[List[str], str],
        raise_error: bool = True,
        input: Optional[str] = None,
        timeout: Optional[int] = None,
        **kwargs,
) -> Tuple[int, str, str]:
    """
    Run shell command in subprocess
    Parameters
    ----------
    cmd: list of str, or str
        Command to execute
    raise_error: bool
        Wheter to raise an error if the command failed
    input: str, optional
        Input string for the command
    timeout: int, optional
        Timeout for the command
    **kwargs:
        Arguments in subprocess.Popen

    Raises
    ------
    AssertionError:
        Raises if the error failed to execute and `raise_error` set to `True`

    Return
    ------
    return_code: int
        The return code of the command
    out: str
        stdout content of the executed command
    err: str
        stderr content of the executed command
    """
    if isinstance(cmd, str):
        cmd = cmd.split()
    elif isinstance(cmd, list):
        cmd = [str(x) for x in cmd]

    sub = subprocess.Popen(
        args=cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        **kwargs
    )
    if input is not None:
        sub.stdin.write(bytes(input, encoding=sys.stdin.encoding))
    try:
        out, err = sub.communicate(timeout=timeout)
        return_code = sub.poll()
    except subprocess.TimeoutExpired:
        sub.kill()
        print("Command %s timeout after %d seconds" % (cmd, timeout))
        return 999, "", ""  # 999 is a special return code for timeout
    out = out.decode(sys.stdin.encoding)
    err = err.decode(sys.stdin.encoding)
    if raise_error and return_code != 0:
        raise CommandExecuteError("Command %s failed: \n%s" % (cmd, err))
    return return_code, out, err

def write_and_submit_to_slurm(cmd,name,dirc,prefix, time = "48:00:00", env = "torch-gpu"):
    print(f"submitting job {name}")
    script_str = f"""#!/bin/bash

#SBATCH --time={time}
#SBATCH --nodes=1
#SBATCH -A lr_ninjaone
#SBATCH -p csd_lr6_192
#SBATCH -q condo_ninjaone
#SBATCH --job-name={prefix}{name}
#SBATCH --output=R-%x.out.txt

source ~/.bashrc
module load python/3.7
conda activate {env}
"""
    script_str += cmd
    cwd = os.getcwd()
    os.chdir(dirc)
    with open(f'{name}.sh', 'w') as f:
        f.write(script_str)
    os.system(f'sbatch {name}.sh')
    os.chdir(cwd)

def is_job_in_queue(jobname):
    result = subprocess.check_output(['squeue', '-u', 'nancy_guan','-o',f'"%.18i %.9P %.{len(jobname)+3}j %.13u %.8T %.10M %.9l %.6D %R"'])
    sqs_str = result.decode('utf-8')
    if jobname in sqs_str:
        return True
    else:
        return False