import os
import socket
import unittest
from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import BatchSampler

try:
    from accelerate.data_loader import BatchSamplerShard
except ImportError:
    BatchSamplerShard = None

from open_r1.grpo_variants import (
    AVSPOState,
    advance_avspo_state,
    apply_avspo_advantages,
    clipped_surrogate,
    compute_avspo_virtual_count,
    compute_ngrpo_advantages,
    find_avspo_collapsed_groups,
    ngrpo_backward_or_arm,
    NGRPOStepGate,
    RepeatGlobalBatchSampler,
    ngrpo_filtered_sequence_loss,
    reduce_avspo_batch_statistics,
)


NUM_GENERATIONS = 8


def _distributed_ngrpo_worker(rank, world_size, port):
    dist.init_process_group(
        "gloo",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=world_size,
    )

    def reduce_fn(value, reduction="sum"):
        result = value.clone()
        dist.all_reduce(result, op=dist.ReduceOp.SUM)
        return result

    sequence_loss = torch.full((8,), 9.0 if rank == 0 else 1.0)
    all_filtered = ngrpo_filtered_sequence_loss(
        sequence_loss,
        torch.tensor([True]),
        8,
        world_size,
        reduce_fn,
    )
    assert all_filtered.skip_update

    partially_filtered = ngrpo_filtered_sequence_loss(
        sequence_loss,
        torch.tensor([rank == 0]),
        8,
        world_size,
        reduce_fn,
    )
    assert not partially_filtered.skip_update
    assert partially_filtered.global_kept_sequences == 8
    mean_loss = partially_filtered.loss.detach().clone()
    dist.all_reduce(mean_loss, op=dist.ReduceOp.SUM)
    assert torch.allclose(mean_loss / world_size, torch.tensor(1.0))
    dist.destroy_process_group()


def grpo_advantages(rewards):
    grouped = rewards.view(-1, NUM_GENERATIONS)
    mean = grouped.mean(dim=1).repeat_interleave(NUM_GENERATIONS)
    std = grouped.std(dim=1).repeat_interleave(NUM_GENERATIONS)
    return (rewards - mean) / (std + 1e-4)


class NGRPOTest(unittest.TestCase):
    @unittest.skipIf(os.name == "nt" or not dist.is_available(), "Gloo fork test requires Linux")
    def test_two_rank_filter_decision_and_global_loss(self):
        sock = socket.socket()
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
        sock.close()
        mp.start_processes(
            _distributed_ngrpo_worker,
            args=(2, port),
            nprocs=2,
            start_method="fork",
            join=True,
        )

    def test_virtual_max_is_added_to_every_group(self):
        rewards = torch.tensor(
            [0.0] * 8
            + [0.5] * 8
            + [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        )
        result = compute_ngrpo_advantages(rewards, NUM_GENERATIONS)

        grouped = rewards.view(-1, NUM_GENERATIONS)
        augmented = torch.cat((grouped, torch.ones((3, 1))), dim=1)
        expected = (grouped - augmented.mean(1, keepdim=True)) / (
            augmented.std(1, keepdim=True) + 1e-6
        )

        self.assertTrue(torch.allclose(result.advantages, expected.flatten()))
        self.assertTrue(torch.all(result.advantages[:8] < 0))
        self.assertTrue(torch.all(result.advantages[8:16] < 0))
        self.assertGreater(result.advantages[16].item(), 0.0)
        self.assertTrue(torch.all(result.advantages[17:] < 0))

    def test_all_max_group_stays_zero(self):
        result = compute_ngrpo_advantages(
            torch.ones(NUM_GENERATIONS), NUM_GENERATIONS
        )
        self.assertTrue(torch.equal(result.advantages, torch.zeros(NUM_GENERATIONS)))
        self.assertTrue(result.all_correct_groups.item())

    def test_author_code_uses_sample_standard_deviation(self):
        result = compute_ngrpo_advantages(
            torch.zeros(NUM_GENERATIONS), NUM_GENERATIONS
        )
        self.assertTrue(
            torch.allclose(
                result.advantages,
                torch.full((NUM_GENERATIONS,), -1.0 / 3.0),
                atol=1e-5,
            )
        )

    def test_task_maximum_must_bound_rewards(self):
        with self.assertRaisesRegex(ValueError, "bound every observed reward"):
            compute_ngrpo_advantages(torch.tensor([0.0, 1.1]), 2)

    def test_asymmetric_clipping_matches_author_code(self):
        ratios = torch.tensor([1.30, 1.00, 0.80, 1.30])
        advantages = torch.tensor([1.0, -1.0, -1.0, -1.0])
        surrogate = clipped_surrogate(ratios, advantages, 0.16, 0.24)
        expected = torch.tensor([1.24, -1.00, -0.84, -1.30])
        self.assertTrue(torch.allclose(surrogate, expected))

    def test_reused_rollout_activates_asymmetric_clipping(self):
        logps = torch.nn.Parameter(torch.zeros(2))
        old_logps = logps.detach().clone()
        advantages = torch.tensor([1.0, -1.0])
        optimizer = torch.optim.SGD([logps], lr=1.0)

        first = -clipped_surrogate(
            torch.exp(logps - old_logps), advantages, 0.16, 0.24
        ).mean()
        first.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        self.assertTrue(torch.allclose(logps, torch.tensor([0.5, -0.5])))

        ratios = torch.exp(logps - old_logps)
        second_surrogate = clipped_surrogate(ratios, advantages, 0.16, 0.24)
        self.assertTrue(torch.allclose(second_surrogate, torch.tensor([1.24, -0.84])))
        (-second_surrogate.mean()).backward()
        self.assertTrue(torch.equal(logps.grad, torch.zeros(2)))

    def test_global_batches_repeat_on_the_same_rank(self):
        def shard(rank):
            sampler = RepeatGlobalBatchSampler(range(8), 2, 2, seed=7)
            return list(sampler)[rank::2]

        rank_zero = shard(0)
        rank_one = shard(1)
        self.assertTrue(all(a == b for a, b in zip(rank_zero[::2], rank_zero[1::2])))
        self.assertTrue(all(a == b for a, b in zip(rank_one[::2], rank_one[1::2])))
        self.assertTrue(all(a != b for a, b in zip(rank_zero[::2], rank_one[::2])))

    @unittest.skipIf(BatchSamplerShard is None, "accelerate is not installed")
    def test_accelerate_shard_preserves_rank_local_reuse(self):
        def shard(rank):
            sampler = RepeatGlobalBatchSampler(range(8), 2, 2, seed=7)
            batches = BatchSampler(sampler, batch_size=1, drop_last=False)
            sharded = BatchSamplerShard(
                batches,
                num_processes=2,
                process_index=rank,
                split_batches=False,
            )
            return [batch[0] for batch in sharded]

        rank_zero = shard(0)
        rank_one = shard(1)
        self.assertTrue(all(a == b for a, b in zip(rank_zero[::2], rank_zero[1::2])))
        self.assertTrue(all(a == b for a, b in zip(rank_one[::2], rank_one[1::2])))
        self.assertTrue(all(a != b for a, b in zip(rank_zero[::2], rank_one[::2])))

    def test_repeat_sampler_is_deterministic_and_epoch_aware(self):
        first = RepeatGlobalBatchSampler(range(12), 4, 2, seed=3)
        second = RepeatGlobalBatchSampler(range(12), 4, 2, seed=3)
        epoch_zero = list(first)
        self.assertEqual(epoch_zero, list(second))
        self.assertEqual(len(epoch_zero), len(first))
        first.set_epoch(1)
        self.assertNotEqual(epoch_zero, list(first))

    def test_filter_uses_global_kept_sequence_denominator(self):
        rank_zero = torch.ones(8, requires_grad=True)
        rank_one = torch.full((8,), 9.0, requires_grad=True)

        def global_eight(local, reduction):
            self.assertEqual(reduction, "sum")
            return local.new_tensor(8.0)

        loss_zero = ngrpo_filtered_sequence_loss(
            rank_zero, torch.tensor([False]), 8, 2, global_eight
        ).loss
        loss_one = ngrpo_filtered_sequence_loss(
            rank_one, torch.tensor([True]), 8, 2, global_eight
        ).loss
        self.assertEqual(((loss_zero + loss_one) / 2).item(), 1.0)

    def test_filter_all_correct_returns_graph_connected_zero(self):
        sequence_loss = torch.ones(8, requires_grad=True)

        def global_zero(local, reduction):
            return local.new_tensor(0.0)

        result = ngrpo_filtered_sequence_loss(
            sequence_loss, torch.tensor([True]), 8, 2, global_zero
        )
        loss = result.loss
        loss.backward()
        self.assertEqual(loss.item(), 0.0)
        self.assertTrue(result.skip_update)
        self.assertEqual(result.global_kept_sequences, 0)
        self.assertTrue(torch.equal(sequence_loss.grad, torch.zeros(8)))

    def test_skip_gate_restores_trainer_step_and_suppresses_events(self):
        gate = NGRPOStepGate()
        state = SimpleNamespace(global_step=4)
        control = SimpleNamespace(
            should_log=True,
            should_save=True,
            should_evaluate=True,
        )

        gate.arm(state.global_step)
        self.assertTrue(gate.before_counter_increment(state))
        self.assertEqual(state.global_step, 3)
        state.global_step += 1
        self.assertTrue(gate.after_counter_increment(state, control))
        self.assertEqual(state.global_step, 4)
        self.assertFalse(control.should_log)
        self.assertFalse(control.should_save)
        self.assertFalse(control.should_evaluate)
        self.assertFalse(gate.armed)

    def test_all_filtered_transaction_preserves_optimizer_state(self):
        parameter = torch.nn.Parameter(torch.tensor(1.0))
        optimizer = torch.optim.AdamW([parameter], lr=0.1, weight_decay=0.0)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)

        class Engine:
            global_steps = 0
            micro_steps = 0

            def backward(self, loss):
                loss.backward()
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                self.global_steps += 1
                self.micro_steps += 1

        engine = Engine()
        gate = NGRPOStepGate()
        state = SimpleNamespace(global_step=0)
        control = SimpleNamespace(
            should_log=True,
            should_save=True,
            should_evaluate=True,
        )

        self.assertTrue(
            ngrpo_backward_or_arm(
                engine.backward, parameter.square(), False, gate, state.global_step
            )
        )
        state.global_step += 1
        adam_state = optimizer.state[parameter]
        snapshot = (
            parameter.detach().clone(),
            adam_state["step"].detach().clone(),
            adam_state["exp_avg"].detach().clone(),
            adam_state["exp_avg_sq"].detach().clone(),
            scheduler.last_epoch,
            scheduler._step_count,
            optimizer.param_groups[0]["lr"],
            engine.global_steps,
            engine.micro_steps,
            state.global_step,
        )

        self.assertFalse(
            ngrpo_backward_or_arm(
                engine.backward, parameter * 0.0, True, gate, state.global_step
            )
        )
        gate.before_counter_increment(state)
        state.global_step += 1
        gate.after_counter_increment(state, control)
        current = (
            parameter.detach(),
            adam_state["step"],
            adam_state["exp_avg"],
            adam_state["exp_avg_sq"],
            scheduler.last_epoch,
            scheduler._step_count,
            optimizer.param_groups[0]["lr"],
            engine.global_steps,
            engine.micro_steps,
            state.global_step,
        )
        for expected, actual in zip(snapshot[:4], current[:4]):
            self.assertTrue(torch.equal(expected, actual))
        self.assertEqual(snapshot[4:], current[4:])

        self.assertTrue(
            ngrpo_backward_or_arm(
                engine.backward, parameter.square(), False, gate, state.global_step
            )
        )
        state.global_step += 1
        self.assertEqual(engine.global_steps, snapshot[7] + 1)
        self.assertEqual(engine.micro_steps, snapshot[8] + 1)
        self.assertEqual(state.global_step, snapshot[9] + 1)


class AVSPOTest(unittest.TestCase):
    def test_stratified_virtual_rewards_cover_both_extremes(self):
        mixed = torch.tensor([1.0, 0.0] * 4)
        rewards = torch.cat((torch.ones(8), torch.zeros(8), mixed))
        baseline = grpo_advantages(rewards)
        result = apply_avspo_advantages(
            baseline,
            rewards,
            NUM_GENERATIONS,
            global_acr=2.0 / 3.0,
            adaptive_threshold=0.5,
        )

        self.assertEqual(result.virtual_count, 7)
        self.assertEqual(result.advantages.numel(), rewards.numel())
        self.assertTrue(torch.equal(result.active_groups, torch.tensor([True, True, False])))
        self.assertTrue(torch.all(result.advantages[:8] > 0))
        self.assertTrue(torch.all(result.advantages[8:16] < 0))
        self.assertTrue(torch.equal(result.advantages[16:], baseline[16:]))
        self.assertTrue(
            torch.allclose(result.virtual_rewards[0], torch.arange(7, 0, -1) / 8)
        )
        self.assertTrue(
            torch.allclose(
                result.virtual_rewards[1],
                0.1 * torch.arange(7, 0, -1) / 7,
            )
        )
        grouped = rewards.view(-1, NUM_GENERATIONS)
        augmented = torch.cat((grouped, result.virtual_rewards), dim=1)
        expected = (grouped - augmented.mean(1, keepdim=True)) / (
            augmented.std(1, keepdim=True, unbiased=False) + 1e-4
        )
        self.assertTrue(torch.allclose(result.advantages[:16], expected[:2].flatten()))

    def test_global_gate_is_strict_and_preserves_grpo(self):
        rewards = torch.cat((torch.ones(8), torch.zeros(8)))
        baseline = grpo_advantages(rewards)
        result = apply_avspo_advantages(
            baseline,
            rewards,
            NUM_GENERATIONS,
            global_acr=1.0,
            adaptive_threshold=1.0,
        )
        self.assertEqual(result.virtual_count, 0)
        self.assertTrue(torch.equal(result.advantages, baseline))
        self.assertFalse(result.active_groups.any().item())

    def test_virtual_count_follows_acr(self):
        self.assertEqual(compute_avspo_virtual_count(0.01, 8), 1)
        self.assertEqual(compute_avspo_virtual_count(2.0 / 3.0, 8), 7)
        self.assertEqual(compute_avspo_virtual_count(1.0, 8), 8)

    def test_collapse_threshold_is_strict(self):
        rewards = torch.tensor([0.0, 2.0, 0.0, 1.0])
        collapsed = find_avspo_collapsed_groups(
            rewards, num_generations=2, collapse_threshold=1.0
        )
        self.assertTrue(torch.equal(collapsed, torch.tensor([False, True])))

    def test_batch_statistics_use_global_reduction(self):
        collapsed = torch.tensor([True])
        rewards = torch.ones(8)

        def fake_reduce(local, reduction):
            self.assertEqual(reduction, "sum")
            return local + torch.tensor([0.0, 1.0, 4.0, 8.0])

        acr, mean_reward = reduce_avspo_batch_statistics(
            collapsed, rewards, fake_reduce
        )
        self.assertEqual(acr, 0.5)
        self.assertEqual(mean_reward, 0.75)

    def test_threshold_state_uses_reward_direction(self):
        initial = advance_avspo_state(AVSPOState(), 0.8, 1.0)
        self.assertEqual(initial.threshold, 0.5)
        self.assertEqual(initial.previous_mean_reward, 1.0)

        improving = advance_avspo_state(initial, 0.8, 2.0)
        degrading = advance_avspo_state(initial, 0.8, 0.0)
        self.assertAlmostEqual(improving.threshold, 0.503)
        self.assertAlmostEqual(degrading.threshold, 0.497)

    def test_threshold_state_is_bounded(self):
        lower = advance_avspo_state(AVSPOState(0.101, 1.0), 1.0, 0.0)
        upper = advance_avspo_state(AVSPOState(0.899, 1.0), 0.0, 0.0)
        self.assertEqual(lower.threshold, 0.1)
        self.assertEqual(upper.threshold, 0.9)

    def test_non_binary_rewards_are_rejected(self):
        rewards = torch.tensor([0.0, 0.5])
        with self.assertRaisesRegex(ValueError, "binary rewards"):
            apply_avspo_advantages(
                torch.zeros_like(rewards),
                rewards,
                2,
                global_acr=1.0,
                adaptive_threshold=0.5,
            )


if __name__ == "__main__":
    unittest.main()
