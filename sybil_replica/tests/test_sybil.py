import numpy as np
import torch

from sybil.model import Cumulative_Probability_Layer, SybilNet
from sybil.phantom import make_phantom
from sybil.predict import hotspots
from sybil.preprocess import apply_windowing, preprocess, to_model_coords
from sybil.train import annotation_loss, concordance_index, survival_labels, survival_loss
from sybil.weights import SimpleIsotonic


def test_forward_shapes_small_volume():
    m = SybilNet().eval()
    with torch.no_grad():
        out = m(torch.randn(1, 3, 32, 64, 64))
    assert out["logit"].shape == (1, 6)
    assert out["image_attention_1"].shape == (1, 4, 16)
    assert out["volume_attention_1"].shape == (1, 4)
    # attentions are log-probabilities
    assert torch.allclose(out["volume_attention_1"].exp().sum(-1), torch.ones(1), atol=1e-5)


def test_cumulative_layer_is_monotone():
    layer = Cumulative_Probability_Layer(16, max_followup=6)
    z = layer(torch.randn(8, 16))
    assert (z[:, 1:] - z[:, :-1] >= -1e-6).all()


def test_state_dict_matches_reference_names():
    keys = set(SybilNet().state_dict())
    for k in ["image_encoder.0.0.weight", "pool.image_pool1.attention_fc.weight", "pool.volume_pool2.conv1d.weight",
              "pool.hidden_fc.weight", "prob_of_failure_layer.upper_triagular_mask",
              "prob_of_failure_layer.base_hazard_fc.bias"]:
        assert k in keys


def test_windowing_matches_reference():
    hu = np.array([-2000, -1350, -600, 150, 3000], np.float32)
    out = apply_windowing(hu) // 256
    assert out[0] == 0 and out[-1] == 255 and 120 <= out[2] <= 135


def test_preprocess_and_coords_hit_the_nodule():
    ct = make_phantom(seed=3, nodule_mm=16)
    prep = preprocess(ct)
    assert prep.tensor.shape == (1, 3, 200, 256, 256)
    z, y, x = to_model_coords(ct, *ct.meta["nodule"]["center_mm"])
    assert prep.display[z, y, x] > 200          # solid nodule is bright in lung window
    assert prep.lung_mask.any()


def test_hotspots_find_peak():
    A = np.zeros((25, 16, 16), np.float32)
    A[10, 4, 12] = 1
    h = hotspots(A)[0]
    assert (h["z"], h["y"], h["x"]) == (84, 72, 200)


def test_survival_labels():
    y, m, c, t = survival_labels(2, 6)
    assert y.tolist() == [0, 0, 1, 1, 1, 1] and m.sum() == 3 and c == 1
    y, m, c, t = survival_labels(None, 3)
    assert y.sum() == 0 and m.tolist() == [1, 1, 1, 1, 0, 0] and c == 0


def test_losses_backprop():
    m = SybilNet()
    out = m(torch.randn(2, 3, 32, 64, 64))
    ann = torch.zeros(2, 1, 200, 256, 256)
    ann[0, :, 90:100, 100:120, 100:120] = 1
    ls = survival_loss(out["logit"], torch.tensor([[0, 0, 1, 1, 1, 1.]] * 2), torch.ones(2, 6))
    la = annotation_loss(out, ann, torch.tensor([1., 0.]))
    (ls + la).backward()
    assert la.item() > 0 and m.pool.image_pool1.attention_fc.weight.grad is not None


def test_concordance_and_isotonic():
    assert concordance_index(np.array([0.9, 0.1]), np.array([1, 5]), np.array([1, 0])) == 1.0
    iso = SimpleIsotonic([[1.0]], [0.0], [0, 1], [0, 0.5])
    assert np.allclose(iso(np.array([0.5])), [0.25])
