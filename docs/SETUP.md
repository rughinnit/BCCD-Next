# Training environment — ROCm 5.4.3 + PyTorch 1.13.1 on gfx803 (RX 570)

Modern ROCm dropped Polaris (gfx803) after 6.1.5, and no official PyTorch ROCm wheel contains
gfx803 kernels. This setup uses a prebuilt community wheel plus a hand-assembled ROCm 5.4.3
user-space from AMD's Ubuntu packages, with no root and no Docker.

## Components

| Path | What |
|---|---|
| `/mnt/Ext/rocm54/root/opt/rocm` | ROCm 5.4.3 user-space (21 packages + roctracer + hip headers + llvm) |
| `/mnt/Ext/rocm54/env.sh` | `ROCM_PATH`, `LD_LIBRARY_PATH`, rocBLAS/MIOpen db paths |
| `/mnt/Ext/sysroot` | OpenMPI 5, OpenBLAS, GCC 13 headers (host-compiled deps torch links against) |
| `/mnt/Ext/venv-tsl` / `venv-tsl2` | Python 3.10 + `torch 1.13.1` + `torchvision 0.14.1` (gfx803 wheels from tsl0922/pytorch-gfx803) |
| `/mnt/Ext/env-tsl.sh` | sources ROCm env + adds sysroot to `LD_LIBRARY_PATH` |

```bash
curl -L -o torch.whl https://github.com/tsl0922/pytorch-gfx803/releases/download/pytorch-1.13.1/torch-1.13.1-cp310-cp310-linux_x86_64.whl
curl -L -o tv.whl    https://github.com/tsl0922/pytorch-gfx803/releases/download/pytorch-1.13.1/torchvision-0.14.1-cp310-cp310-linux_x86_64.whl
uv venv --python 3.10 /mnt/Ext/venv-tsl
uv pip install --python /mnt/Ext/venv-tsl/bin/python torch.whl tv.whl
uv pip install --python /mnt/Ext/venv-tsl/bin/python "numpy<2" ultralytics
```

## Three fixes that are mandatory

1. **exec-stack flag.** ROCm 5.4.3 ships `libamdhip64.so.5` / `libhiprtc.so.5` with
   `PT_GNU_STACK` marked executable; glibc ≥ 2.41 refuses to `dlopen` them. Clear `PF_X` on the
   `PT_GNU_STACK` program header in place (backups kept). Without it, `import torch` fails with
   `cannot enable executable stack as shared object requires`.
2. **`libmpi_cxx.so.40` stub.** torch 1.13 links OpenMPI's C++ bindings, which OpenMPI 5 removed.
   A 5-symbol stub (`MPI::Comm::Comm`, `MPI::Datatype::Free`, `MPI::Win::Free`,
   `ompi_mpi_cxx_op_intercept`, `ompi_op_set_cxx_callback`) satisfies the loader. MPI is never
   initialised.
3. **`numpy<2`.** torch 1.13 was compiled against NumPy 1.x.

## MIGraphX (optional, faster inference)

MIGraphX 2.4.0 from the same ROCm 5.4.3 repo (8 MB) plus `rocm-llvm` (794 MB) for device JIT.
It needs `/opt/rocm` and `/opt/rocm-5.4.3` to exist, which is done with `bwrap`:

```bash
bwrap --dev-bind / / --tmpfs /opt \
      --bind /mnt/Ext/rocm54/root/opt/rocm-5.4.3 /opt/rocm-5.4.3 \
      --bind /mnt/Ext/rocm54/root/opt/rocm-5.4.3 /opt/rocm \
      /mnt/Ext/rocm54/root/opt/rocm/bin/migraphx-driver perf model.onnx
```

ROCm's clang 15 cannot parse Arch's GCC 16 headers, so `llvm/bin/clang++` is a wrapper adding
`-isystem` paths to GCC 13 headers extracted from the Arch archive.

MIGraphX 2.4.0's ONNX parser rejects the opset-13 `Split` form: **export with `opset=12`**.

## Verification before trusting training

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_arch_list())"
# 1.13.1 True ['gfx803']
```

Numerics checked against CPU: GEMM 1.3e-06, conv2d 5.5e-07, BatchNorm exact, LayerNorm 4.8e-07
(the known gfx803 LayerNorm bug does **not** affect this ROCm 5.4.3 build).
