"""Worker threads have no dependency on Streamlit state or APIs."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from queue import Queue
from .. import audit, CancellationToken


@dataclass
class AuditJob:
    token: CancellationToken = field(default_factory=CancellationToken)
    events: Queue = field(default_factory=Queue)
    future: object = None
    executor: object = None

    @classmethod
    def start(cls, data, **config):
        job = cls()
        job.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="data-audit")
        job.future = job.executor.submit(
            audit,
            data.X,
            data.y,
            groups=data.groups,
            metadata=data.metadata,
            cancellation_token=job.token,
            progress=job.events.put,
            **config,
        )
        job.future.add_done_callback(lambda _: job.executor.shutdown(wait=False))
        return job
