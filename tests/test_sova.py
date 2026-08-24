import unittest

import torch

from open_r1.sova import apply_sova_advantages, validate_sova_configuration


NUM_GENERATIONS = 8


def baseline_advantages(total_rewards):
    grouped = total_rewards.view(-1, NUM_GENERATIONS)
    mean = grouped.mean(dim=1).repeat_interleave(NUM_GENERATIONS)
    std = grouped.std(dim=1).repeat_interleave(NUM_GENERATIONS)
    return (total_rewards - mean) / (std + 1e-4)


def apply(mode, total, accuracy, format_rewards, lambda_positive, lambda_negative):
    baseline = baseline_advantages(total)
    return baseline, apply_sova_advantages(
        baseline_advantages=baseline,
        total_rewards=total,
        accuracy_rewards=accuracy,
        format_rewards=format_rewards,
        num_generations=NUM_GENERATIONS,
        mode=mode,
        lambda_positive=lambda_positive,
        lambda_negative=lambda_negative,
    )


class SOVAModeTest(unittest.TestCase):
    def test_negative_mode_reproduces_all_wrong_virtual_max(self):
        accuracy = torch.zeros(8)
        format_rewards = torch.ones(8)
        total = accuracy + format_rewards

        baseline, result = apply(
            "negative", total, accuracy, format_rewards, 0.0, 0.125
        )

        self.assertTrue(torch.equal(baseline, torch.zeros(8)))
        self.assertTrue(result.negative_active_groups.item())
        self.assertFalse(result.positive_active_groups.item())
        self.assertAlmostEqual(result.negative_effective_mean.item(), 10 / 9, places=6)
        self.assertAlmostEqual(result.negative_effective_std.item(), 1 / 3, places=6)
        self.assertTrue(
            torch.allclose(result.advantages, torch.full((8,), -0.04165417))
        )

    def test_full_negative_branch_matches_original_sova_statistics(self):
        accuracy = torch.tensor(
            [0.0] * 8 + [1.0, 0.5, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0]
        )
        format_rewards = torch.tensor([1.0, 0.0] * 4 + [1.0] * 8)
        total = accuracy + format_rewards
        baseline = baseline_advantages(total)

        grouped = total.view(-1, 8)
        grouped_accuracy = accuracy.view(-1, 8)
        base_mean = grouped.mean(dim=1)
        base_std = grouped.std(dim=1)
        active = grouped_accuracy.eq(0).all(dim=1)
        augmented = torch.cat((grouped, torch.full((2, 1), 2.0)), dim=1)
        expected_mean = torch.where(active, augmented.mean(dim=1), base_mean)
        expected_std = torch.where(active, augmented.std(dim=1), base_std)
        expected = (total - expected_mean.repeat_interleave(8)) / (
            expected_std.repeat_interleave(8) + 1e-4
        )

        result = apply_sova_advantages(
            baseline,
            total,
            accuracy,
            format_rewards,
            8,
            "negative",
            0.0,
            1.0,
        )

        self.assertTrue(torch.equal(result.negative_active_groups, active))
        self.assertTrue(torch.equal(result.advantages, expected))

    def test_positive_mode_reinforces_only_fully_correct_formatted_group(self):
        accuracy = torch.ones(8)
        format_rewards = torch.ones(8)
        total = accuracy + format_rewards

        baseline, result = apply(
            "positive", total, accuracy, format_rewards, 0.0625, 0.0
        )

        self.assertTrue(torch.equal(baseline, torch.zeros(8)))
        self.assertTrue(result.positive_active_groups.item())
        self.assertFalse(result.negative_active_groups.item())
        self.assertAlmostEqual(result.positive_effective_mean.item(), 17.5 / 9, places=6)
        self.assertAlmostEqual(result.positive_effective_std.item(), 1 / 6, places=6)
        self.assertTrue(
            torch.allclose(result.advantages, torch.full((8,), 0.02082084))
        )

    def test_positive_gate_rejects_mixed_or_missing_format_reward(self):
        accuracy = torch.ones(16)
        format_rewards = torch.tensor([1.0, 0.0] * 4 + [0.0] * 8)
        total = accuracy + format_rewards

        baseline, result = apply(
            "positive", total, accuracy, format_rewards, 0.0625, 0.0
        )

        self.assertTrue(torch.equal(result.positive_active_groups, torch.zeros(2, dtype=torch.bool)))
        self.assertTrue(torch.equal(result.advantages, baseline))

    def test_negative_gate_preserves_format_order_in_mixed_format_group(self):
        accuracy = torch.zeros(8)
        format_rewards = torch.tensor([1.0, 0.0] * 4)
        total = accuracy + format_rewards

        baseline, result = apply(
            "negative", total, accuracy, format_rewards, 0.0, 0.125
        )

        self.assertTrue(result.negative_active_groups.item())
        self.assertTrue(torch.all(result.advantages[::2] > result.advantages[1::2]))
        self.assertTrue(torch.all(result.advantages[::2] > 0))
        self.assertTrue(torch.all(result.advantages[1::2] < 0))
        self.assertFalse(torch.equal(result.advantages, baseline))

    def test_negative_gate_accepts_all_zero_format_group(self):
        accuracy = torch.zeros(8)
        format_rewards = torch.zeros(8)
        total = accuracy + format_rewards

        baseline, result = apply(
            "negative", total, accuracy, format_rewards, 0.0, 0.125
        )

        self.assertTrue(result.negative_active_groups.item())
        self.assertTrue(torch.equal(baseline, torch.zeros(8)))
        self.assertTrue(torch.all(result.advantages < 0))

    def test_bidirectional_routes_positive_negative_and_inactive_groups(self):
        positive_accuracy = torch.ones(8)
        positive_format = torch.ones(8)
        negative_accuracy = torch.zeros(8)
        negative_format = torch.ones(8)
        informative_accuracy = torch.tensor([1.0, 0.5, 0.0, 0.5, 1.0, 0.0, 0.5, 0.0])
        informative_format = torch.ones(8)
        accuracy = torch.cat(
            (positive_accuracy, negative_accuracy, informative_accuracy)
        )
        format_rewards = torch.cat(
            (positive_format, negative_format, informative_format)
        )
        total = accuracy + format_rewards

        baseline, result = apply(
            "bidirectional", total, accuracy, format_rewards, 0.0625, 0.03125
        )

        self.assertTrue(
            torch.equal(
                result.positive_active_groups,
                torch.tensor([True, False, False]),
            )
        )
        self.assertTrue(
            torch.equal(
                result.negative_active_groups,
                torch.tensor([False, True, False]),
            )
        )
        self.assertTrue(torch.all(result.advantages[:8] > 0))
        self.assertTrue(torch.all(result.advantages[8:16] < 0))
        self.assertTrue(torch.equal(result.advantages[16:], baseline[16:]))
        self.assertEqual(result.advantages.shape, total.shape)

    def test_one_sided_modes_do_not_apply_the_disabled_branch(self):
        accuracy = torch.cat((torch.ones(8), torch.zeros(8)))
        format_rewards = torch.ones(16)
        total = accuracy + format_rewards

        baseline, positive = apply(
            "positive", total, accuracy, format_rewards, 0.0625, 0.0
        )
        _, negative = apply(
            "negative", total, accuracy, format_rewards, 0.0, 0.03125
        )

        self.assertTrue(torch.equal(positive.advantages[8:], baseline[8:]))
        self.assertTrue(torch.equal(negative.advantages[:8], baseline[:8]))

    def test_uniform_partial_and_mixed_groups_are_bitwise_baseline(self):
        accuracy = torch.cat(
            (
                torch.full((8,), 0.5),
                torch.tensor([1.0, 0.0, 0.5, 0.0, 1.0, 0.5, 0.0, 0.5]),
            )
        )
        format_rewards = torch.ones(16)
        total = accuracy + format_rewards

        baseline, result = apply(
            "bidirectional", total, accuracy, format_rewards, 0.0625, 0.03125
        )

        self.assertTrue(torch.equal(result.advantages, baseline))
        self.assertFalse(result.positive_active_groups.any().item())
        self.assertFalse(result.negative_active_groups.any().item())

    def test_exact_gate_boundaries_do_not_use_tolerances(self):
        accuracy = torch.cat(
            (
                torch.full((8,), 1e-6),
                torch.full((8,), 1.0 - 1e-6),
                torch.ones(8),
            )
        )
        format_rewards = torch.cat(
            (
                torch.ones(16),
                torch.full((8,), 1.0 - 1e-6),
            )
        )
        total = accuracy + format_rewards

        baseline, result = apply(
            "bidirectional", total, accuracy, format_rewards, 0.0625, 0.03125
        )

        self.assertFalse(result.positive_active_groups.any().item())
        self.assertFalse(result.negative_active_groups.any().item())
        self.assertTrue(torch.equal(result.advantages, baseline))


class SOVAValidationTest(unittest.TestCase):
    def test_valid_mode_matrix(self):
        validate_sova_configuration("positive", 0.0625, 0.0)
        validate_sova_configuration("negative", 0.0, 0.125)
        validate_sova_configuration("bidirectional", 0.0625, 0.03125)

    def test_invalid_mode_or_lambda_matrix_fails_closed(self):
        invalid = [
            ("both", 0.1, 0.1),
            ("positive", 0.0, 0.0),
            ("positive", 0.1, 0.1),
            ("negative", 0.1, 0.1),
            ("negative", 0.0, 0.0),
            ("bidirectional", 0.1, 0.0),
            ("bidirectional", 0.0, 0.1),
            ("bidirectional", float("nan"), 0.1),
            ("bidirectional", 0.1, float("inf")),
            ("bidirectional", 1.01, 0.1),
            ("bidirectional", 0.1, -0.01),
        ]
        for mode, lambda_positive, lambda_negative in invalid:
            with self.subTest(
                mode=mode,
                lambda_positive=lambda_positive,
                lambda_negative=lambda_negative,
            ), self.assertRaises(ValueError):
                validate_sova_configuration(
                    mode, lambda_positive, lambda_negative
                )

    def test_directionally_invalid_virtual_rewards_fail_closed(self):
        with self.assertRaises(ValueError):
            validate_sova_configuration(
                "positive", 0.1, 0.0, positive_virtual_total_reward=2.0
            )
        with self.assertRaises(ValueError):
            validate_sova_configuration(
                "negative", 0.0, 0.1, negative_virtual_total_reward=1.0
            )

    def test_invalid_shapes_and_nonfinite_rewards_fail_closed(self):
        baseline = torch.zeros(8)
        common = dict(
            baseline_advantages=baseline,
            total_rewards=torch.ones(8),
            accuracy_rewards=torch.zeros(8),
            format_rewards=torch.ones(8),
            num_generations=8,
            mode="negative",
            lambda_positive=0.0,
            lambda_negative=0.125,
        )
        for replacement in (
            {"total_rewards": torch.ones(7)},
            {"accuracy_rewards": torch.zeros(2, 4)},
            {"num_generations": 1},
            {"num_generations": 4},
            {"total_rewards": torch.tensor([float("nan")] + [1.0] * 7)},
            {"accuracy_rewards": torch.full((8,), -0.1)},
            {"format_rewards": torch.full((8,), 1.1)},
            {"total_rewards": torch.full((8,), 1.5)},
        ):
            kwargs = {**common, **replacement}
            with self.subTest(replacement=replacement), self.assertRaises(ValueError):
                apply_sova_advantages(**kwargs)

    def test_lambda_endpoints_and_invalid_eps(self):
        validate_sova_configuration("positive", 1.0, 0.0)
        validate_sova_configuration("negative", 0.0, 1.0)
        common = dict(
            baseline_advantages=torch.zeros(8),
            total_rewards=torch.ones(8),
            accuracy_rewards=torch.zeros(8),
            format_rewards=torch.ones(8),
            num_generations=8,
            mode="negative",
            lambda_positive=0.0,
            lambda_negative=0.125,
        )
        for eps in (0.0, float("nan"), float("inf")):
            with self.subTest(eps=eps), self.assertRaises(ValueError):
                apply_sova_advantages(**common, eps=eps)


if __name__ == "__main__":
    unittest.main()
