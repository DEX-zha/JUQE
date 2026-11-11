/// Pack dimensions returned by the C ABI.
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct PackInfo {
    /// Number of layers, including dense placeholders.
    pub layers: u32,
    /// Experts per routed layer.
    pub experts: u32,
}

/// Quantized matrix geometry for one expert.
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct PartInfo {
    /// Opaque ggml type code, validated by the engine adapter.
    pub kind: u32,
    /// Matrix dimensions.
    pub dimensions: [u64; 2],
    /// Bytes occupied by one matrix.
    pub bytes: u64,
}

/// One layer's packed expert layout.
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct LayerInfo {
    /// Packed bytes per expert; zero denotes a dense layer.
    pub stride: u64,
    /// Gate, up, and down matrix geometry.
    pub parts: [PartInfo; 3],
}

/// Engine-owned contiguous storage for one matrix across all slots.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct PoolPart {
    /// Writable pool base; must remain stable until runtime destruction.
    pub data: *mut u8,
    /// Accessible allocation length, including any engine-owned zero slot.
    pub bytes: u64,
}

/// Destination pools for one layer.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct LayerPool {
    /// Writable expert slots; excludes the engine's optional zero slot.
    pub slots: u32,
    /// Gate, up, and down pools. Dense layers use null pointers and zero lengths.
    pub parts: [PoolPart; 3],
}

/// C ABI resource configuration.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct RuntimeConfig {
    /// I/O worker count.
    pub workers: u32,
    /// Maximum waiting jobs.
    pub queue_capacity: u32,
    /// Zero selects buffered I/O; one selects direct scatter I/O.
    pub direct: u32,
}

/// Counters returned after a complete successful batch preparation.
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct BatchInfo {
    /// Distinct resident experts.
    pub cache_hits: u64,
    /// Distinct loaded experts.
    pub loads: u64,
    /// Actual read bytes.
    pub bytes_read: u64,
    /// Submission through publication, in seconds.
    pub request_to_ready_seconds: f64,
}
