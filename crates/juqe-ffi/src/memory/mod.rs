mod region;

use crate::{LayerPool, boundary::invalid};
use juqe_core::format::ExpertIndex;
use juqe_io::{AlignedBuffer, ExpertReader, OpenMode, ReadReport};
use juqe_runtime::ExpertStorage;
use region::Region;
use std::{io, path::Path, sync::Mutex, time::Instant};

type Slot = Mutex<[Region; 3]>;

pub(crate) struct MemoryStorage {
    index: ExpertIndex,
    reader: ExpertReader,
    layers: Vec<Vec<Slot>>,
}

impl MemoryStorage {
    /// The engine must keep pools live and follow the runtime's batch lease protocol.
    pub unsafe fn new(
        index: ExpertIndex,
        path: &Path,
        mode: OpenMode,
        pools: &[LayerPool],
    ) -> io::Result<Self> {
        region::validate(&index, pools)?;
        let reader = ExpertReader::open(path, &index, mode)?;
        let mut layers = Vec::with_capacity(pools.len());
        for (layout, pool) in index.layers().iter().zip(pools) {
            let mut slots = Vec::with_capacity(pool.slots as usize);
            for slot in 0..pool.slots as usize {
                let mut regions = Vec::with_capacity(3);
                for (matrix, part) in layout.parts().iter().zip(&pool.parts) {
                    let bytes = matrix.bytes() as usize;
                    // SAFETY: validation covers all slot extents and pool disjointness.
                    // The engine retains each allocation until the runtime is closed.
                    let region = unsafe { Region::new(part.data.add(slot * bytes), bytes)? };
                    regions.push(region);
                }
                let regions = regions
                    .try_into()
                    .map_err(|_| invalid("expected three pool regions"))?;
                slots.push(Mutex::new(regions));
            }
            layers.push(slots);
        }
        Ok(Self {
            index,
            reader,
            layers,
        })
    }
}

impl ExpertStorage for MemoryStorage {
    fn load(
        &self,
        layer: usize,
        expert: usize,
        slot: usize,
        scratch: &mut AlignedBuffer,
        requested_at: Instant,
    ) -> io::Result<ReadReport> {
        let destination = self
            .layers
            .get(layer)
            .and_then(|slots| slots.get(slot))
            .ok_or_else(|| invalid("destination slot out of range"))?;
        let mut destination = destination
            .lock()
            .map_err(|_| io::Error::other("destination lock poisoned"))?;
        let [gate, up, down] = &mut *destination;
        self.reader.read_into(
            &self.index.layers()[layer],
            expert,
            [gate.as_mut_slice(), up.as_mut_slice(), down.as_mut_slice()],
            scratch,
            requested_at,
        )
    }
}
