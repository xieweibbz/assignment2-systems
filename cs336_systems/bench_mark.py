import cs336_basics.model
import cs336_basics.nn_utils
import cs336_basics.optimizer
import cs336_basics
import torch
import time


def bench_mark(batch_size, vocab_size, context_length, d_model, num_layers, num_heads, d_ff, rope_theta, warmup_steps, num_steps):
    print("Benchmarking BasicsTransformerLM")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Generate tensor with size batch_size x context_length, values between 0 and vocab_size
    input = torch.randint(0, vocab_size, size=(batch_size, context_length), device=device)
    target = input.clone()
    
    model = cs336_basics.model.BasicsTransformerLM(
        vocab_size=vocab_size,
        context_length=context_length,
        d_model=512,
        num_layers=6,
        num_heads=8,
        d_ff=2048,
        rope_theta=10000.0)
    model.to(device)
    optimizer = cs336_basics.optimizer.AdamW(model.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)

    for _ in range(warmup_steps):
        print(f"Warmup step {_ + 1} of {warmup_steps}")
        optimizer.zero_grad()
        logits = model(input)
        loss = cs336_basics.nn_utils.cross_entropy(logits, target)
        loss.backward()
        optimizer.step()
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    fw_time = 0
    bw_time = 0
    opt_time = 0
    for _ in range(num_steps):
        print(f"Step {_ + 1} of {num_steps}")
        optimizer.zero_grad()
        start_time = time.time()
        logits = model(input)
        loss = cs336_basics.nn_utils.cross_entropy(logits, target)
        fw_end_time = time.time()
        loss.backward()
        bw_end_time = time.time()
        optimizer.step()
        opt_end_time = time.time()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        fw_time += fw_end_time - start_time
        bw_time += bw_end_time - fw_end_time
        opt_time += opt_end_time - bw_end_time

    fw_time /= num_steps
    bw_time /= num_steps
    opt_time /= num_steps

    return (fw_time, bw_time, opt_time)


if __name__ == "__main__":
    result = bench_mark(batch_size=4, vocab_size=10000, context_length=8, d_model=512, num_layers=6, num_heads=8, d_ff=2048, rope_theta=10000.0, warmup_steps=0, num_steps=10)
    print(f"Time forward taken: {result[0]} seconds")
    print(f"Time backward taken: {result[1]} seconds")
    print(f"Time optimizer step taken: {result[2]} seconds")