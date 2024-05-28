import torch

import pytest

from einops import rearrange

from causal_conv1d.causal_conv1d_interface import causal_conv1d_fn
from causal_conv1d.causal_conv1d_interface import causal_conv1d_update


def time_(fn, name, steps=500):
    steps = 500
    # warm up
    for _ in range(100):
        fn()

    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(steps)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(steps)]

    for i in range(steps):
        start_events[i].record()
        fn()
        end_events[i].record()

    torch.cuda.synchronize()
    times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
    print(f'Time {name}: {sum(times) / steps} ms')


def bench_causal_conv1d(dim, seqlen, width, has_bias, silu_activation, itype, channel_last, has_initial_states, return_final_states):
    if not channel_last and (has_initial_states or return_final_states):
        pytest.skip("Only channel_last support initial_states or return_final_states")
    device = "cuda"
    # set seed
    torch.random.manual_seed(0)
    batch = 4
    if not channel_last:
        x = torch.randn(batch, 4096 + dim + 64, seqlen, device=device, dtype=itype)[:, 4096:4096 + dim, :].requires_grad_()
    else:
        x = rearrange(
            torch.randn(batch, seqlen, 4096 + dim + 64, device=device, dtype=itype)[:, :, 4096:4096 + dim], "b s d -> b d s"
        ).requires_grad_()
    weight = torch.randn(dim, width, device=device, dtype=torch.float32, requires_grad=True)
    if has_bias:
        bias = torch.randn(dim, device=device, dtype=torch.float32, requires_grad=True)
    else:
        bias = None
    if has_initial_states:
        initial_states = torch.randn(batch, width - 1, dim, device=device, dtype=itype).transpose(1, 2).requires_grad_()
    else:
        initial_states = None
    activation = None if not silu_activation else "silu"

    cc_fwd = lambda: causal_conv1d_fn(x, weight, bias, initial_states=initial_states, return_final_states=return_final_states,
                                      activation=activation)
    time_(cc_fwd, 'causal conv fwd')

    steps = 500
    for _ in range(100):
        out = causal_conv1d_fn(x, weight, bias, initial_states=initial_states, return_final_states=return_final_states,
                           activation=activation)
        g = torch.randn_like(out)
        out.backward(g)

    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(steps)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(steps)]

    for i in range(steps):
        out = causal_conv1d_fn(x, weight, bias, initial_states=initial_states, return_final_states=return_final_states,
                           activation=activation)
        g = torch.randn_like(out)
        start_events[i].record()
        out.backward(g)
        end_events[i].record()

    torch.cuda.synchronize()
    times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
    print(f'Time causal conv bwd:', sum(times) / steps)

def bench_causal_conv1d_update(dim, width, has_bias, silu_activation, itype):
    device = "cuda"
    # set seed
    torch.random.manual_seed(0)
    batch = 4
    # batch = 1
    # dim = 64
    x = torch.randn(batch, dim, device=device, dtype=itype)
    conv_state = torch.randn(batch, dim, width, device=device, dtype=itype)
    weight = torch.randn(dim, width, device=device, dtype=torch.float32, requires_grad=True)
    if has_bias:
        bias = torch.randn(dim, device=device, dtype=torch.float32, requires_grad=True)
    else:
        bias = None
    activation = None if not silu_activation else "silu"

    ccu = lambda: causal_conv1d_update(x, conv_state, weight, bias, activation=activation)
    time_(ccu, 'causal conv update')

bench_causal_conv1d(5376, 4096, 4, True, True, torch.float16, True, False, False)
bench_causal_conv1d_update(5376, 4, True, True, torch.float16)

