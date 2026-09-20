(** Page-locked (pinned) host memory.

    A copy out of ordinary pageable memory cannot be a DMA: the driver
    stages it through a page-locked bounce buffer, which halves the
    bandwidth and makes an "async" copy synchronous with respect to the
    host. A [Value] whose storage came from here is page-locked, so the
    copy is a straight DMA and {!Buffer.upload_async} really does return
    before the bytes have moved.

    cudajit 0.7 does not bind [cuMemHostAlloc], so this module calls the
    driver directly through ctypes. The library is opened lazily: on a
    machine with no [libcuda.so.1] the module still loads, and only a
    call fails. *)

open Cudacaml_ir

(** A host [Value] in page-locked memory from [cuMemHostAlloc] with
    [CU_MEMHOSTALLOC_PORTABLE], zero-filled like {!Value.create}. Freed by
    a finaliser through [cuMemFreeHost]; the Bigarray inside the [Value]
    is what keeps the allocation alive, so the memory outlives every copy
    that can still reach it. [Failure] if the driver refuses, and
    [Invalid_argument] for [Bool]. *)
val alloc : 'a Dtype.t -> Shape.t -> 'a Value.t

(** Copy of an ordinary [Value] into pinned memory. The original is
    untouched and the two do not share storage. *)
val of_value : 'a Value.t -> 'a Value.t

(** True if this [Value]'s storage came from {!alloc} or {!of_value}.
    Tracked by the address of the allocation in a table the finaliser
    prunes, so an address is never reported as pinned after it has been
    handed back to the driver. *)
val is_pinned : _ Value.t -> bool
