import unittest
from unittest.mock import Mock, patch
from contextlib import redirect_stdout
from io import StringIO

import numpy as np
import torch

from neural_encoder.utils import AveragingLogger, ConsoleLogger, RunningAverages, WandbLogger
from neural_encoder.nonlinear import NonlinearRefiner


class MemoryLogger(AveragingLogger):
    def __init__(self, log_frequency=3):
        super().__init__(log_frequency)
        self.records = []
        self.closed = 0

    def _write(self, metrics, *, step):
        self.records.append((step, dict(metrics)))

    def _finish(self):
        self.closed += 1


class TestLogging(unittest.TestCase):
    def test_running_averages_scalar_tensors_sparse_keys_and_clear(self):
        values = RunningAverages()
        tensor = torch.tensor(2., requires_grad=True)
        values.update("loss", tensor)
        values.update("loss", 4)
        values.update("accuracy", 0.8)
        self.assertEqual(values.get("loss"), 3)
        self.assertEqual(values.get("missing"), 0)
        self.assertEqual(values.get_and_clear_all(), {"loss": 3., "accuracy": 0.8})
        self.assertEqual(values.get_and_clear_all(), {})
        self.assertIsNone(tensor.grad)
        values.update("new", 7)
        values.clear("new")
        self.assertEqual(list(values), [])

    def test_manual_accumulation_frequency_forced_dump_and_finish(self):
        logger = MemoryLogger(log_frequency=2)
        value = torch.tensor(1., requires_grad=True)
        self.assertIs(logger.logkv("loss", value), value)
        logger.logkv("loss", 3)
        logger.dumpkvs()
        self.assertEqual(logger.records, [])
        logger.logkvs({"loss": 5, "val": 9})
        logger.dumpkvs()
        self.assertEqual(logger.records, [(2, {"loss": 3., "val": 9.})])
        logger.logkv("loss", 8)
        logger.dumpkvs(force=True)
        self.assertEqual(logger.records[-1], (3, {"loss": 8.}))
        logger.log({"loss": 6}, step=4)
        logger.finish()
        logger.finish()
        self.assertEqual(logger.records[-1], (4, {"loss": 6.}))
        self.assertEqual(logger.log_step, 4)
        self.assertEqual(logger.closed, 1)

    def test_explicit_steps_console_and_disabled_wandb(self):
        stream = StringIO()
        with redirect_stdout(stream):
            logger = ConsoleLogger(log_frequency=2)
            logger.log({"loss": 2}, step=10)
            self.assertEqual(stream.getvalue(), "")
            logger.log({"loss": 4}, step=20)
        self.assertIn("step=20 loss=3", stream.getvalue())
        with patch.dict("sys.modules", {"wandb": None}):
            logger = WandbLogger(mode="disabled", log_frequency=2)
            self.assertIsNone(logger.wandb_run)
            with self.assertLogs(level="INFO") as logs:
                logger.log({"loss": 2}, step=1)
                logger.finish()
            self.assertIn("step=1", logs.output[0])

    def test_wandb_backend_mocked(self):
        wandb = Mock()
        with patch.dict("sys.modules", {"wandb": wandb}), patch.dict(
            "os.environ", {"WANDB_PROJECT": "test-project", "WANDB_ENTITY": "test-entity"},
        ):
            logger = WandbLogger(mode="offline", log_frequency=2, name="run")
            self.assertEqual(wandb.init.call_args.kwargs["project"], "test-project")
            self.assertEqual(wandb.init.call_args.kwargs["mode"], "offline")
            logger.log({"loss": 1}, step=1)
            logger.log({"loss": 3}, step=2)
            wandb.init.return_value.log.assert_called_once_with({"loss": 2.}, step=2)
            logger.finish()
            wandb.init.return_value.finish.assert_called_once()

    def test_refiner_flushes_without_closing(self):
        logger = MemoryLogger(log_frequency=2)
        views = [np.random.default_rng(i).normal(size=(6, 3)) for i in range(2)]
        model = NonlinearRefiner(
            network_kwargs={"hidden_dim": 4, "dropout": 0}, steps=3,
            batch_size=4, device="cpu", random_state=0, logger=logger,
        ).fit_views(views)
        self.assertEqual([step for step, _ in logger.records], [2, 3])
        self.assertAlmostEqual(logger.records[0][1]["loss"],
                               (model.history_[0]["loss"] + model.history_[1]["loss"]) / 2)
        self.assertEqual(logger.closed, 0)
        self.assertEqual(len(model.history_), 3)
        logger.finish()
        self.assertEqual(len(logger.records), 2)


if __name__ == "__main__":
    unittest.main()
