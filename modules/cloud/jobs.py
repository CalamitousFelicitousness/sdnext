"""Cloud job manager.

Each call to `submit_job(capability, provider_id, request)` spawns a daemon
thread that opens its own asyncio event loop, runs the provider handler to
completion (or cancellation), and updates the in-memory `JOBS` dict. Terminal
jobs are evicted to `HISTORY` (LRU bounded by `cloud_job_history_size`).

The runner handles two provider modes:

  - mode='sync':  awaits handler['predict'](request). If predict is not a
                  coroutine function it is wrapped with asyncio.to_thread.
  - mode='async': calls handler['submit'](job), then loops calling
                  handler['poll'](job) every poll_interval seconds until the
                  job reaches a terminal status. Between polls, the runner
                  observes shared.state.interrupted, the per-job
                  cancel_requested flag, and the cloud_job_max_duration
                  watchdog. On any cancellation trigger it dispatches
                  handler['cancel'](job) fire-and-forget and marks the job
                  cancelled.

Progress events are published via modules.api.ws.publish (lazy import) so the
cloud framework remains usable in environments where the FastAPI app has not
been started.

Known limitation: a sync-mode provider whose `predict` hangs indefinitely will
hang its Job thread. The sync UI wrapper uses its own outer watchdog
(`cloud_job_max_duration + 30s`) to recover. If hangs become common, switch the
runner's sync path to asyncio.wait_for with best-effort task cancellation.
"""
from __future__ import annotations
import asyncio
import collections
import inspect
import threading
import time
import uuid
from typing import Optional
from modules.logger import log
from modules.cloud.types import Job, TERMINAL_JOB_STATUSES
from modules.cloud.registry import get_handler


JOBS: dict[str, Job] = {}
HISTORY: 'collections.OrderedDict[str, Job]' = collections.OrderedDict()
JOBS_LOCK = threading.Lock()
BACKGROUND_TASKS: set = set()


DEFAULT_POLL_INTERVAL = 5.0
DEFAULT_MAX_DURATION = 600.0
DEFAULT_HISTORY_SIZE = 50


def _opt(name: str, default):
    try:
        from modules import shared  # pylint: disable=import-outside-toplevel
        value = getattr(shared.opts, name, default)
        return default if value is None else value
    except Exception:
        return default


def _now() -> float:
    return time.time()


def _interrupted() -> bool:
    try:
        from modules import shared  # pylint: disable=import-outside-toplevel
        return bool(shared.state.interrupted)
    except Exception:
        return False


def _state_begin(job: Job) -> Optional[int]:
    try:
        from modules import shared  # pylint: disable=import-outside-toplevel
        return shared.state.begin(title=f'cloud:{job.capability}:{job.provider_id}', api=True)
    except Exception:
        return None


def _state_end(task_id: Optional[int]) -> None:
    try:
        from modules import shared  # pylint: disable=import-outside-toplevel
        shared.state.end(task_id=task_id, api=True)
    except Exception:
        pass


def _publish(job: Job, kind: str) -> None:
    try:
        from modules.api import ws  # pylint: disable=import-outside-toplevel
    except Exception:
        return
    event = {
        'type': kind,
        'job_id': job.id,
        'provider_id': job.provider_id,
        'capability': job.capability,
        'status': job.status,
        'progress': job.progress,
        'message': job.message,
        'error': job.error,
        'ts': job.updated_at or _now(),
    }
    try:
        ws.publish(event)
    except Exception as e:
        log.debug(f'Cloud: publish failed: {e}')


def _touch(job: Job) -> None:
    job.updated_at = _now()


def _evict_to_history(job: Job) -> None:
    with JOBS_LOCK:
        JOBS.pop(job.id, None)
        HISTORY[job.id] = job
        max_size = int(_opt('cloud_job_history_size', DEFAULT_HISTORY_SIZE))
        while len(HISTORY) > max_size:
            HISTORY.popitem(last=False)


def submit_job(capability: str, provider_id: str, request) -> Job:
    job = Job(
        id=uuid.uuid4().hex,
        provider_id=provider_id,
        capability=capability,
        status='pending',
        started_at=_now(),
        updated_at=_now(),
        request=request,
    )
    handler = get_handler(capability, provider_id)
    if handler is None:
        job.status = 'failed'
        job.error = f'Unknown {capability} provider: {provider_id}'
        _touch(job)
        with JOBS_LOCK:
            HISTORY[job.id] = job
        _publish(job, 'cloud.job.terminal')
        return job
    with JOBS_LOCK:
        JOBS[job.id] = job
    _publish(job, 'cloud.job.created')
    threading.Thread(target=_run, args=(job, handler), name=f'cloud-job-{job.id[:8]}', daemon=True).start()
    return job


def get_job(job_id: str) -> Optional[Job]:
    with JOBS_LOCK:
        return JOBS.get(job_id) or HISTORY.get(job_id)


def list_jobs(*, capability: Optional[str] = None, status: Optional[str] = None) -> list[Job]:
    with JOBS_LOCK:
        all_jobs = list(JOBS.values()) + list(HISTORY.values())
    if capability is not None:
        all_jobs = [j for j in all_jobs if j.capability == capability]
    if status is not None:
        all_jobs = [j for j in all_jobs if j.status == status]
    all_jobs.sort(key=lambda j: j.started_at, reverse=True)
    return all_jobs


def cancel_job(job_id: str) -> bool:
    with JOBS_LOCK:
        job = JOBS.get(job_id)
    if job is None:
        return False
    if job.status in TERMINAL_JOB_STATUSES:
        return False
    job.cancel_requested = True
    return True


def _run(job: Job, handler: dict) -> None:
    loop = asyncio.new_event_loop()
    task_id = _state_begin(job)
    try:
        asyncio.set_event_loop(loop)
        loop.run_until_complete(_runner(job, handler))
    except Exception as e:
        log.error(f'Cloud job runner crashed: id={job.id} provider={job.provider_id} error={e}')
        if job.status not in TERMINAL_JOB_STATUSES:
            job.status = 'failed'
            job.error = str(e)
            _touch(job)
            _publish(job, 'cloud.job.terminal')
    finally:
        try:
            loop.close()
        except Exception:
            pass
        _state_end(task_id)
        _evict_to_history(job)


async def _runner(job: Job, handler: dict) -> None:
    mode = handler.get('mode', 'sync')
    if mode == 'sync':
        await _run_sync(job, handler)
    elif mode == 'async':
        await _run_async(job, handler)
    else:
        job.status = 'failed'
        job.error = f'Unknown handler mode: {mode!r}'
        _touch(job)
        _publish(job, 'cloud.job.terminal')


async def _run_sync(job: Job, handler: dict) -> None:
    job.status = 'running'
    job.progress = 0.05
    _touch(job)
    _publish(job, 'cloud.job.progress')

    predict = handler.get('predict')
    if predict is None:
        job.status = 'failed'
        job.error = "handler missing 'predict' for mode='sync'"
        _touch(job)
        _publish(job, 'cloud.job.terminal')
        return
    try:
        if inspect.iscoroutinefunction(predict):
            result = await predict(job.request)
        else:
            result = await asyncio.to_thread(predict, job.request)
        job.result = result
        if getattr(result, 'error', None):
            job.status = 'failed'
            job.error = result.error
        else:
            job.status = 'succeeded'
            job.progress = 1.0
    except Exception as e:
        job.status = 'failed'
        job.error = str(e)
    finally:
        _touch(job)
        _publish(job, 'cloud.job.terminal')


async def _run_async(job: Job, handler: dict) -> None:
    submit = handler.get('submit')
    poll = handler.get('poll')
    cancel = handler.get('cancel')
    poll_interval = handler.get('poll_interval') or float(_opt('cloud_job_poll_default', DEFAULT_POLL_INTERVAL))
    max_duration = float(_opt('cloud_job_max_duration', DEFAULT_MAX_DURATION))

    job.status = 'running'
    _touch(job)
    _publish(job, 'cloud.job.progress')
    try:
        await submit(job)
    except Exception as e:
        job.status = 'failed'
        job.error = f'submit failed: {e}'
        _touch(job)
        _publish(job, 'cloud.job.terminal')
        return
    if not job.status or job.status == 'pending':
        job.status = 'submitted'
    _touch(job)
    _publish(job, 'cloud.job.progress')

    while job.status not in TERMINAL_JOB_STATUSES:
        if job.cancel_requested or _interrupted() or (_now() - job.started_at) > max_duration:
            job.status = 'cancelled'
            if not job.message:
                job.message = 'cancelled'
            _touch(job)
            _publish(job, 'cloud.job.terminal')
            if cancel is not None:
                try:
                    cancel_task = asyncio.ensure_future(cancel(job))
                    BACKGROUND_TASKS.add(cancel_task)
                    cancel_task.add_done_callback(BACKGROUND_TASKS.discard)
                except Exception as e:
                    log.warning(f'Cloud: cancel dispatch failed for {job.provider_id}: {e}')
            return
        try:
            await poll(job)
        except Exception as e:
            job.status = 'failed'
            job.error = f'poll failed: {e}'
            _touch(job)
            _publish(job, 'cloud.job.terminal')
            return
        _touch(job)
        if job.status in TERMINAL_JOB_STATUSES:
            _publish(job, 'cloud.job.terminal')
            return
        _publish(job, 'cloud.job.progress')
        await asyncio.sleep(poll_interval)
