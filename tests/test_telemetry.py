import importlib.util
import unittest


@unittest.skipUnless(importlib.util.find_spec("torch"), "Install torch")
class TelemetryTests(unittest.TestCase):
    def test_nonfinite_loss_rejected(self):
        from decision_lab.telemetry import TrainingTelemetry
        telemetry = TrainingTelemetry("cpu")
        for loss in (float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                telemetry.step(loss)

    def test_step_aggregation_on_cpu(self):
        from decision_lab.telemetry import TrainingTelemetry
        telemetry = TrainingTelemetry("cpu")
        telemetry.step(2.0)
        telemetry.step(1.0)
        result = telemetry.finish()
        self.assertEqual(result["microsteps"], 2)
        self.assertEqual(result["first_50_loss"], 1.5)
        self.assertGreaterEqual(result["training_seconds"], result["step_p50_seconds"])
        self.assertIsNone(result["cuda_peak_allocated_bytes"])
