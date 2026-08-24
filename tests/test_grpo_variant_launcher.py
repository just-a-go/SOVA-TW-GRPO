from pathlib import Path
import re
import unittest


SCRIPT = Path(__file__).parents[1] / "scripts" / "sova-tw-grpo.sh"


class GRPOVariantLauncherTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.script = SCRIPT.read_text(encoding="utf-8")

    def test_loss_selector_lists_all_methods(self):
        self.assertIn("LOSS_TYPE: tw_grpo | grpo | ngrpo | avspo", self.script)
        self.assertIn('grpo|ngrpo|avspo)', self.script)

    def test_variants_use_binary_reward_without_sova(self):
        block = re.search(
            r"grpo\|ngrpo\|avspo\)(.*?)\n\s*;;",
            self.script,
            re.DOTALL,
        )
        self.assertIsNotNone(block)
        self.assertIn("USE_SOVA=false", block.group(1))
        self.assertIn("REWARD_FUNCS=(origin_accuracy)", block.group(1))

    def test_tw_grpo_keeps_sova_reward_pair(self):
        block = re.search(r"tw_grpo\)(.*?)\n\s*;;", self.script, re.DOTALL)
        self.assertIsNotNone(block)
        self.assertRegex(block.group(1), r"USE_SOVA=(true|false)")
        self.assertIn("REWARD_FUNCS=(accuracy format)", block.group(1))

    def test_tw_grpo_without_sova_uses_baseline_run_name(self):
        run_block = re.search(
            r"# Run identity and outputs.*?case \"\$\{LOSS_TYPE\}\" in(.*?)\n\s*ngrpo\)",
            self.script,
            re.DOTALL,
        )
        self.assertIsNotNone(run_block)
        self.assertIn('if [[ "${USE_SOVA}" == "true" ]]', run_block.group(1))
        self.assertIn(
            'MODEL_RUN="Qwen2.5-VL-7B-Instruct_clevrer_counterfactual_twgrpo_a1p70"',
            run_block.group(1),
        )

    def test_only_sova_receives_sova_arguments(self):
        self.assertIn(
            'METHOD_ARGS=(--loss_type "${LOSS_TYPE}" --use_sova "${USE_SOVA}")',
            self.script,
        )
        self.assertRegex(
            self.script,
            r'if \[\[ "\$\{USE_SOVA\}" == "true" \]\]; then\s+METHOD_ARGS\+=\(',
        )

    def test_ngrpo_run_name_discloses_core_scope(self):
        self.assertIn(
            "ngrpo_core_reuse${NGRPO_NUM_ITERATIONS}_drop_norefill_rmax1p0",
            self.script,
        )

    def test_ngrpo_enables_rollout_reuse(self):
        self.assertIn("NGRPO_NUM_ITERATIONS=2", self.script)
        self.assertIn(
            'METHOD_ARGS+=(--ngrpo_num_iterations "${NGRPO_NUM_ITERATIONS}")',
            self.script,
        )
        self.assertIn("NGRPO requires reuse >= 2", self.script)


if __name__ == "__main__":
    unittest.main()
