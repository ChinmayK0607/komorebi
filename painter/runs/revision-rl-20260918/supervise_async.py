"""Own the trainer/inference process group and stop it if the judge stalls."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import uuid


def atomic(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    temp.replace(path)


def stop(process):
    if process is None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=10)


def main(root):
    if sys.platform != 'linux':
        raise SystemExit('Training supervisor runs on the node only')
    experiment = json.loads((root/'EXPERIMENT.json').read_text())
    if experiment['judge'] != 'gpt-5.6-luna' or experiment.get('judge_reasoning_effort') != 'low':
        raise ValueError('This restart requires actual Luna low agents')
    sys.path.insert(0, str(root))
    from training_handoff import write_runtime_state
    train = None
    reason = 'interrupted'
    code = 1
    def interrupted(signum, frame):
        raise KeyboardInterrupt()
    signal.signal(signal.SIGTERM, interrupted)
    try:
        train_env = dict(os.environ)
        train_env.pop('OPENROUTER_API_KEY', None)
        train = subprocess.Popen(['bash', str(root/'launch_async.sh'), str(root)],
                                 env=train_env, start_new_session=True)
        atomic(root/'supervisor-status.json', dict(status='running', started=time.time(),
                                                  training_pid=train.pid, judge='gpt-5.6-luna',
                                                  reasoning_effort='low', judge_mode='external_agents'))
        write_runtime_state(root, launch_pid=train.pid)
        pending_since = None
        last_progress = time.time()
        previous_completed = 0
        while train.poll() is None:
            requests = list((root/'judge-queue').glob('*/request.json'))
            completed = sum(f.with_name('receipt.json').exists() for f in requests)
            pending = len(requests) - completed
            if completed > previous_completed:
                last_progress = time.time()
            previous_completed = completed
            atomic(root/'judge-service-status.json', dict(at=time.time(), judge_model='gpt-5.6-luna',
                reasoning_effort='low', mode='external_agents', pending=pending,
                completed=completed, last_success=last_progress))
            write_runtime_state(root, launch_pid=train.pid)
            if pending:
                pending_since = pending_since or time.time()
                if time.time() - max(pending_since, last_progress) > 900:
                    raise RuntimeError('Luna rewards made no progress on pending work for 15 minutes')
            else:
                pending_since = None
            time.sleep(5)
        code = train.returncode
        reason = 'training_completed' if code == 0 else 'training_failed'
    except (Exception, KeyboardInterrupt) as exc:
        reason = str(exc) or 'interrupted'
    finally:
        stop(train)
        atomic(root/'supervisor-status.json', dict(status='stopped', reason=reason, exit_code=code,
                                                  stopped=time.time()))
        write_runtime_state(root, launch_pid=train.pid if train is not None else None)
        print('Supervisor stopped:', reason, flush=True)
    return code


if __name__ == '__main__':
    raise SystemExit(main(Path(sys.argv[1]).resolve()))
