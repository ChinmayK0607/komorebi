"""One real canvas revision, native traced sampling, asynchronous Luna rewards."""
import asyncio
import base64
import hashlib
import json
import os
import sys
import time
import uuid
from pathlib import Path

import verifiers.v1 as vf

RENDER_LIMIT = asyncio.Semaphore(8)
JUDGE_MODEL = os.environ.get('PAINTING_JUDGE_MODEL', 'gpt-5.6-luna')


def atomic(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    temp.replace(path)


def create_or_verify_request(path, value):
    """Install one queue request exactly once and verify cache hits.

    The request is content-addressed by its id, but its creation time is the
    queue's FIFO age.  Replacing it on every identical rollout made a cache hit
    look new and could race a judge assignment or receipt.  ``link`` gives us
    an atomic no-clobber install on the local filesystem; a pre-existing file
    is accepted only when its immutable request fields still match.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    expected = dict(value)
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    temporary.write_text(json.dumps(expected, indent=2) + '\n')
    try:
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.is_symlink() or not path.is_file():
                raise ValueError('Judge request path is not a regular file: ' + str(path))
            try:
                existing = json.loads(path.read_text())
            except json.JSONDecodeError as exc:
                raise ValueError('Existing judge request is invalid JSON: ' + str(path)) from exc
            immutable = ('id', 'image_sha256', 'protocol')
            if any(existing.get(key) != expected.get(key) for key in immutable):
                raise ValueError('Existing judge request conflicts with the content-addressed request: ' + str(path))
            # Same image-conditioned key may be reached from a different
            # rollout directory.  Keep the first paths, but verify their
            # bytes still match the immutable hashes before reusing the queue.
            root = path.parent.parent.parent
            for role, digest in existing.get('image_sha256', {}).items():
                source = Path(existing.get('images', {}).get(role, ''))
                if not source.is_absolute():
                    source = root / source
                if not source.is_file() or sha(source) != digest:
                    raise ValueError('Existing judge request image changed: ' + str(source))
            # Preserve the first timestamp and all queue state.  A request with
            # a missing legacy timestamp remains valid and is not rewritten.
            return existing
        return expected
    finally:
        temporary.unlink(missing_ok=True)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class PainterRevisionData(vf.TaskData):
    root: str
    state: dict


class PainterRevisionTask(vf.Task[PainterRevisionData]):
    @vf.stop
    async def single_turn(self, trace: vf.Trace) -> bool:
        return trace.num_turns >= 1

    @vf.reward(weight=1.0)
    async def visual_quality(self, trace: vf.Trace) -> float:
        if trace.is_truncated:
            raise ValueError('Operational completion cutoff; excluded from learning')
        root = Path(self.data.root)
        # Task scoring may run in a worker process whose import path does not
        # inherit the taskset loader's path.  Make the node workspace explicit
        # before importing the renderer and reward protocol.
        sys.path.insert(0, str(root))
        from render_actions import one
        from quality_signal import reconcile, reward
        state = self.data.state
        folder = root/'rl-rollouts'/trace.id
        sample = dict(id=trace.id, state=state, text=trace.last_reply,
                      finish_reason='stop', trace_id=trace.id,
                      num_output_tokens=trace.num_output_tokens)
        atomic(folder/'sample.json', sample)
        async with RENDER_LIMIT:
            result = await asyncio.to_thread(one, folder/'sample.json')
        trace.info.update(painting_result=result, sample_path=str(folder/'sample.json'))
        if not result['valid']:
            score = reward(result, None)
            if score is None:
                raise ValueError('Excluded operational failure: ' + result.get('error','unknown'))
            raw = json.dumps(result, sort_keys=True, separators=(',', ':')).encode()
            evidence = dict(status='scored', request_id='local-render-' + trace.id,
                            receipt_sha256=hashlib.sha256(raw).hexdigest(), timeout=False, approval=False)
            atomic(folder/'reward.json', dict(judgment=None, reward=score,
                request_id=evidence['request_id'], score_evidence=evidence))
            return score
        paths = dict(reference=state['reference'], anchor=state['canvas'], candidate=result['canvas'])
        hashes = {k:sha(root/v) for k,v in paths.items()}
        key = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
        queue = root/'judge-queue'/key
        # Reward cache is image-conditioned and independent of policy/version.
        # Each trace retains its exact raw response and native sampled logprobs.
        create_or_verify_request(queue/'request.json', dict(id=key, images=paths, image_sha256=hashes,
            created_at=time.time(), protocol='luna-ab-ba-v1'))
        trace.info['judge_request_id'] = key
        started = time.monotonic()
        while not (queue/'receipt.json').is_file():
            if time.monotonic()-started > 1800:
                raise TimeoutError('Luna reward pending; excluded, never converted into a zero reward')
            await asyncio.sleep(1)
        receipt = json.loads((queue/'receipt.json').read_text())
        if (receipt['id'] != key or receipt['image_sha256'] != hashes
                or receipt['judge_model'] != JUDGE_MODEL or receipt.get('reasoning_effort') != 'low'):
            raise ValueError('Luna receipt provenance mismatch')
        judgment = reconcile(receipt.get('candidate_A'), receipt.get('candidate_B'))
        score = reward(result, judgment)
        evidence = dict(status='scored' if score is not None else 'excluded', request_id=key,
                        receipt_sha256=sha(queue/'receipt.json'), timeout=False, approval=False)
        trace.info.update(judge_receipt=receipt, visual_reward=score,
                          judge_wait_seconds=time.monotonic()-started)
        atomic(folder/'reward.json', dict(judgment=judgment, reward=score, request_id=key,
            score_evidence=evidence))
        if score is None:
            raise ValueError('Uncertain or order-inconsistent Luna labels; excluded from learning')
        return score


class PainterRevisionConfig(vf.TasksetConfig):
    root: str


class PainterRevisionTaskset(vf.Taskset[PainterRevisionTask, PainterRevisionConfig]):
    def load(self) -> list[PainterRevisionTask]:
        root = Path(self.config.root).resolve()
        sys.path.insert(0,str(root))
        from contract import SYSTEM,NEXT_VERSION
        def image(path):
            url='data:image/png;base64,'+base64.b64encode(path.read_bytes()).decode()
            return vf.ImageUrlContentPart(image_url=vf.ImageUrlSource(url=url))
        states=json.loads((root/'rl-states.json').read_text())
        experiment=json.loads((root/'EXPERIMENT.json').read_text())
        assert not set(experiment['training_state_groups']) & set(experiment['heldout_groups'])
        return [PainterRevisionTask(PainterRevisionData(idx=i,root=str(root),state=state,
            system_prompt=SYSTEM,prompt=[vf.UserMessage(content=[
                vf.TextContentPart(text='REFERENCE'), image(root/state['reference']),
                vf.TextContentPart(text='CURRENT CANVAS'),image(root/state['canvas']),
                vf.TextContentPart(text='Current program:\n'+(root/state['program']).read_text()),
                vf.TextContentPart(text=NEXT_VERSION)])]),self.config.task)
            for i,state in enumerate(states)]
