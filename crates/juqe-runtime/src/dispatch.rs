use crate::{RuntimeError, batch::BatchReport, pool::WorkerPool, worker::Job};
use juqe_core::cache::{Cache, Lease, LoadTicket};
use std::time::Instant;

pub(crate) fn load(
    pool: &mut WorkerPool,
    cache: &mut Cache,
    layer: usize,
    tickets: Vec<LoadTicket>,
    leases: &mut [Option<Lease>],
    requested_at: Instant,
    report: &mut BatchReport,
) -> Result<(), RuntimeError> {
    let mut submitted = 0;
    let mut failure = None;
    let mut tickets = tickets.into_iter();
    while let Some(ticket) = tickets.next() {
        let job = Job {
            layer,
            ticket,
            requested_at,
        };
        if let Err(job) = pool.submit(job) {
            // These jobs never reached a worker; release only their reservations.
            let _ = cache.fail(&job.ticket);
            for ticket in tickets {
                let _ = cache.fail(&ticket);
            }
            failure = Some(RuntimeError::Stopped);
            break;
        }
        submitted += 1;
    }
    // Always drain submitted jobs, including after the first failure. Their writes
    // must end before this function releases any successful leases.
    for _ in 0..submitted {
        let completion = match pool.receive() {
            Ok(completion) => completion,
            Err(error) => {
                pool.shutdown();
                return Err(RuntimeError::Worker(error));
            }
        };
        let expert = completion.job.ticket.expert();
        let outcome = match completion.result {
            Ok(read) => {
                report.loads += 1;
                report.bytes_read += read.bytes_read;
                cache
                    .complete(&completion.job.ticket)
                    .map(|lease| leases[expert] = Some(lease))
                    .map_err(|source| RuntimeError::Cache { layer, source })
            }
            Err(source) => {
                let _ = cache.fail(&completion.job.ticket);
                Err(RuntimeError::Load {
                    layer,
                    expert,
                    source,
                })
            }
        };
        if failure.is_none() {
            failure = outcome.err();
        }
    }
    match failure {
        Some(error) => Err(error),
        None => Ok(()),
    }
}
