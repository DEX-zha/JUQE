use crate::{
    Config, ExpertStorage,
    worker::{self, Completion, Job},
};
use juqe_core::format::PACK_ALIGNMENT;
use juqe_io::AlignedBuffer;
use std::{
    io,
    sync::{
        Arc, Mutex,
        mpsc::{self, Receiver, SyncSender},
    },
    thread::{self, JoinHandle},
};

pub(crate) struct WorkerPool {
    jobs: Option<SyncSender<Job>>,
    completed: Receiver<Completion>,
    workers: Vec<JoinHandle<()>>,
}

impl WorkerPool {
    pub fn new(
        config: &Config,
        scratch_bytes: usize,
        storage: Arc<dyn ExpertStorage>,
    ) -> io::Result<Self> {
        let (jobs, receiver) = mpsc::sync_channel(config.queue_capacity);
        let (completed, results) = mpsc::channel();
        let receiver = Arc::new(Mutex::new(receiver));
        let mut pool = Self {
            jobs: Some(jobs),
            completed: results,
            workers: Vec::new(),
        };
        for worker_id in 0..config.workers {
            let scratch = AlignedBuffer::new(scratch_bytes, PACK_ALIGNMENT as usize)?;
            let jobs = Arc::clone(&receiver);
            let completed = completed.clone();
            let storage = Arc::clone(&storage);
            let worker = thread::Builder::new()
                .name(format!("juqe-io-{worker_id}"))
                .spawn(move || worker::run(jobs, completed, storage, scratch))?;
            pool.workers.push(worker);
        }
        Ok(pool)
    }

    pub fn submit(&self, job: Job) -> Result<(), Job> {
        match &self.jobs {
            Some(sender) => sender.send(job).map_err(|error| error.0),
            None => Err(job),
        }
    }

    pub fn receive(&self) -> io::Result<Completion> {
        self.completed
            .recv()
            .map_err(|_| io::Error::new(io::ErrorKind::BrokenPipe, "all expert workers stopped"))
    }

    pub fn shutdown(&mut self) {
        // Closing submissions lets workers drain queued writes before joining.
        self.jobs.take();
        for worker in self.workers.drain(..) {
            let _ = worker.join();
        }
    }
}

impl Drop for WorkerPool {
    fn drop(&mut self) {
        self.shutdown();
    }
}
