use crate::ExpertStorage;
use juqe_core::cache::LoadTicket;
use juqe_io::{AlignedBuffer, ReadReport};
use std::{
    io,
    panic::{AssertUnwindSafe, catch_unwind},
    sync::{
        Arc, Mutex,
        mpsc::{Receiver, Sender},
    },
    time::Instant,
};

pub(crate) struct Job {
    pub layer: usize,
    pub ticket: LoadTicket,
    pub requested_at: Instant,
}

pub(crate) struct Completion {
    pub job: Job,
    pub result: io::Result<ReadReport>,
}

pub(crate) fn run(
    jobs: Arc<Mutex<Receiver<Job>>>,
    completed: Sender<Completion>,
    storage: Arc<dyn ExpertStorage>,
    mut scratch: AlignedBuffer,
) {
    loop {
        let job = match jobs.lock() {
            Ok(receiver) => match receiver.recv() {
                Ok(job) => job,
                Err(_) => return,
            },
            Err(_) => return,
        };
        let result = catch_unwind(AssertUnwindSafe(|| {
            storage.load(
                job.layer,
                job.ticket.expert(),
                job.ticket.slot(),
                &mut scratch,
                job.requested_at,
            )
        }))
        .unwrap_or_else(|_| Err(io::Error::other("storage loader panicked")));
        if completed.send(Completion { job, result }).is_err() {
            return;
        }
    }
}
