import numpy as np
import torch

from pycanc.model import Cumulative_Probability_Layer, PyCancNet
from pycanc.phantom import make_phantom
from pycanc.predict import hotspots
from pycanc.preprocess import apply_windowing, preprocess, to_model_coords
from pycanc.train import annotation_loss, concordance_index, survival_labels, survival_loss
from pycanc.weights import SimpleIsotonic


def test_forward_shapes_small_volume():
    m = PyCancNet().eval()
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
    keys = set(PyCancNet().state_dict())
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
    m = PyCancNet()
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


def test_plcom2012_reference_profile_and_monotone():
    from pycanc.clinical import Patient, plcom2012
    base = plcom2012(Patient(age=62, smoking_status="former", cigarettes_per_day=24.87, smoking_years=27,
                             years_since_quit=10, education="some_college", bmi=27))
    # at the model's centring values only the intercept (and the cig term ~0) remains
    assert abs(base["logit"] - (-4.532506)) < 0.01
    older = plcom2012(Patient(age=72, smoking_status="current", smoking_years=50, copd=True))
    assert older["risk"] > base["risk"] and older["eligible"]
    assert plcom2012(Patient(age=60, smoking_status="never"))["risk"] is None


def test_tta_augment_shapes_and_identity():
    from pycanc.predict import augment
    x = torch.randn(1, 3, 4, 32, 32)
    assert torch.equal(augment(x, 0, 0, 0), x)
    y = augment(x, 2, 2, 5)
    assert y.shape == x.shape and torch.equal(y[:, 0], y[:, 1])


def test_metrics_and_isotonic_fit():
    from pycanc.evaluate import auc, isotonic_fit
    assert auc(np.array([0.9, 0.8, 0.1, 0.2]), np.array([1, 1, 0, 0])) == 1.0
    assert auc(np.array([0.5, 0.5]), np.array([1, 0])) == 0.5
    x0, y0 = isotonic_fit(np.array([0.1, 0.2, 0.3, 0.4]), np.array([0, 1, 0, 1]))
    assert np.all(np.diff(y0) >= 0)


def test_fusion_logistic_recovers_signal():
    from pycanc.fusion import fit_logistic
    rng = np.random.default_rng(0)
    X = rng.normal(size=(400, 2))
    y = (rng.random(400) < 1 / (1 + np.exp(-(0.5 + 2 * X[:, 0])))).astype(float)
    w = fit_logistic(X, y)
    assert 1.2 < w[1] < 2.8 and abs(w[2]) < 0.5


def test_quality_checks_flag_thick_slices():
    from pycanc.preprocess import CTVolume, quality_checks
    ct = CTVolume(np.full((40, 64, 64), -800, np.float32), (5.0, 6.0, 6.0))
    ct.hu[:, 20:40, 20:40] = 40
    assert any("Slice spacing" in w for w in quality_checks(ct))
