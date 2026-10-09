"""Select physical nvidia-smi GPUs by UUID, independent of CUDA's device order."""

import subprocess


def gpu_status() -> dict[int, tuple[str, int, int]]:
    """Return {physical_index: (UUID, total MiB, used MiB)} in one query."""
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,uuid,memory.total,memory.used", "--format=csv,noheader,nounits"],
        check=True, capture_output=True, text=True,
    )
    devices = {}
    for line in result.stdout.splitlines():
        index, uuid, total, used = (part.strip() for part in line.split(","))
        devices[int(index)] = (uuid, int(total), int(used))
    return devices


def available_uuids(ids: list[int], max_used_mib: int, min_total_mib: int = 0) -> list[str] | None:
    devices = gpu_status()
    if any(index not in devices for index in ids):
        raise ValueError(f"Unknown physical GPU in {ids}")
    if any(devices[index][1] < min_total_mib for index in ids):
        raise ValueError(f"Selected GPU has less than {min_total_mib} MiB VRAM: {ids}")
    if any(devices[index][2] > max_used_mib for index in ids):
        return None
    return [devices[index][0] for index in ids]
