import unittest

import torch

from optimus.modeling.dense import DenseGaussianPatcher, dense_noise_tensor
from randopt_lora_lab.lora_space import Candidate


class TinyAttention(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = torch.nn.Linear(5, 4, bias=False)
        self.v_proj = torch.nn.Linear(5, 3, bias=False)
        self.o_proj = torch.nn.Linear(4, 5, bias=False)


class TinyLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = TinyAttention()


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Module()
        self.model.layers = torch.nn.ModuleList([TinyLayer()])


class DenseSpaceTests(unittest.TestCase):
    def test_dense_noise_uses_canonical_module_name(self):
        candidate = Candidate("dense_gaussian", seed=123, sigma=0.01)

        bare = dense_noise_tensor("model.layers.0.self_attn.q_proj", (4, 5), candidate)
        peft = dense_noise_tensor("base_model.model.model.layers.0.self_attn.q_proj", (4, 5), candidate)

        self.assertTrue(torch.equal(bare, peft))

    def test_patcher_restores_base_weights(self):
        model = TinyModel()
        patcher = DenseGaussianPatcher(model, ("q_proj", "v_proj"))
        original = {name: module.weight.detach().clone() for name, module in model.named_modules() if name in patcher.module_names}

        patcher.set_candidate(Candidate("dense_gaussian", seed=456, sigma=0.02))
        self.assertFalse(torch.equal(model.model.layers[0].self_attn.q_proj.weight, original["model.layers.0.self_attn.q_proj"]))

        patcher.clear()

        for name, module in model.named_modules():
            if name in original:
                self.assertTrue(torch.equal(module.weight, original[name]))

    def test_same_candidate_is_deterministic_after_restore(self):
        model = TinyModel()
        patcher = DenseGaussianPatcher(model, ("q_proj", "v_proj"))
        candidate = Candidate("dense_gaussian", seed=789, sigma=0.03)

        patcher.set_candidate(candidate)
        first = {name: module.weight.detach().clone() for name, module in model.named_modules() if name in patcher.module_names}
        patcher.clear()
        patcher.set_candidate(candidate)

        for name, module in model.named_modules():
            if name in first:
                self.assertTrue(torch.equal(module.weight, first[name]))

    def test_antithetic_sign_flips_dense_delta(self):
        model = TinyModel()
        patcher = DenseGaussianPatcher(model, ("q_proj",))
        name = "model.layers.0.self_attn.q_proj"
        base = model.model.layers[0].self_attn.q_proj.weight.detach().clone()

        patcher.set_candidate(Candidate("dense_gaussian", seed=2468, sigma=0.01, sign=1))
        positive_delta = model.model.layers[0].self_attn.q_proj.weight.detach().clone() - base
        patcher.clear()
        patcher.set_candidate(Candidate("dense_gaussian", seed=2468, sigma=0.01, sign=-1))
        negative_delta = model.model.layers[0].self_attn.q_proj.weight.detach().clone() - base

        self.assertTrue(torch.allclose(positive_delta, -negative_delta))
        self.assertIn(name, patcher.module_names)

    def test_all_params_mode_patches_every_floating_parameter(self):
        model = TinyModel()
        patcher = DenseGaussianPatcher(model, ("all_params",))
        original = {name: param.detach().clone() for name, param in model.named_parameters()}

        patcher.set_candidate(Candidate("dense_gaussian", seed=8642, sigma=0.01))

        self.assertEqual(set(patcher.module_names), set(original))
        for name, param in model.named_parameters():
            self.assertFalse(torch.equal(param, original[name]))

    def test_upstream_noise_mode_uses_same_seed_stream_per_parameter(self):
        candidate = Candidate("dense_gaussian", seed=123, sigma=0.01)

        first = dense_noise_tensor("a", (2, 2), candidate, noise_mode="upstream")
        second = dense_noise_tensor("b", (2, 2), candidate, noise_mode="upstream")
        upstream = dense_noise_tensor("b", (2, 2), candidate, noise_mode="upstream")
        canonical = dense_noise_tensor("b", (2, 2), candidate, noise_mode="canonical")

        self.assertTrue(torch.equal(first, second))
        self.assertTrue(torch.equal(first, upstream))
        self.assertFalse(torch.equal(first, canonical))


if __name__ == "__main__":
    unittest.main()
