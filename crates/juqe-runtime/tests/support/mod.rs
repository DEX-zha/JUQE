use juqe_core::format::ExpertIndex;
use juqe_io::{AlignedBuffer, ExpertReader, OpenMode, ReadReport};
use juqe_runtime::{Config, ExpertStorage, Runtime};
use std::{
    fs, io,
    path::PathBuf,
    sync::{
        Arc, Mutex,
        atomic::{AtomicUsize, Ordering},
    },
    time::Instant,
};

static NEXT: AtomicUsize = AtomicUsize::new(0);

pub struct Fixture {
    pub path: PathBuf,
    pub storage: Arc<Storage>,
    pub bytes: Vec<u8>,
}

pub struct Storage {
    pub index: ExpertIndex,
    reader: ExpertReader,
    slots: Vec<Mutex<[Vec<u8>; 3]>>,
    pub loads: AtomicUsize,
}

impl Fixture {
    pub fn new(slots: usize) -> Self {
        let path = std::env::temp_dir().join(format!(
            "juqe-runtime-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&path).unwrap();
        let index = "1 4\n0 0 4096 0 128 1 0 512 0 128 1 1024 512 0 128 1 2560 512\n"
            .parse::<ExpertIndex>()
            .unwrap();
        let bytes = (0..16384)
            .map(|offset| (offset % 251) as u8)
            .collect::<Vec<_>>();
        fs::write(path.join("experts.bin"), &bytes).unwrap();
        let reader =
            ExpertReader::open(path.join("experts.bin"), &index, OpenMode::Buffered).unwrap();
        let storage = Arc::new(Storage {
            index,
            reader,
            slots: (0..slots)
                .map(|_| Mutex::new(std::array::from_fn(|_| vec![0; 512])))
                .collect(),
            loads: AtomicUsize::new(0),
        });
        Self {
            path,
            storage,
            bytes,
        }
    }

    pub fn runtime(&self, slots: usize) -> Runtime {
        Runtime::new(
            self.storage.index.clone(),
            Config {
                workers: 2,
                queue_capacity: 1,
                layer_slots: vec![slots],
            },
            self.storage.clone(),
        )
        .unwrap()
    }

    pub fn check(&self, slot: usize, expert: usize) {
        let matrices = self.storage.slots[slot].lock().unwrap();
        for (part, matrix) in self.storage.index.layers()[0]
            .parts()
            .iter()
            .zip(matrices.iter())
        {
            let start = expert * 4096 + part.offset() as usize;
            assert_eq!(matrix, &self.bytes[start..start + 512]);
        }
    }
}

impl ExpertStorage for Storage {
    fn load(
        &self,
        layer: usize,
        expert: usize,
        slot: usize,
        scratch: &mut AlignedBuffer,
        requested_at: Instant,
    ) -> io::Result<ReadReport> {
        self.loads.fetch_add(1, Ordering::Relaxed);
        let mut matrices = self.slots[slot].lock().unwrap();
        let [gate, up, down] = &mut *matrices;
        self.reader.read_into(
            &self.index.layers()[layer],
            expert,
            [gate, up, down],
            scratch,
            requested_at,
        )
    }
}

impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.path);
    }
}
