//! Risk inference: runs the embedded ONNX model
//! (`assets/risk_model.onnx`, trained on the Alibaba DSN'21 SSD dataset)
//! and produces an explainable result.
//!
//! The exported graph wins a model-selection step between logistic
//! regression and a histogram gradient-boosting ensemble (see
//! `assets/train_metrics.json`). Explanations always come from the logistic
//! coefficients in `assets/risk_coefficients.json`; when the exported model
//! is the tree ensemble those contributions are an approximation
//! (`approx_explain: true` in the JSON).

use serde::Serialize;

use crate::features::FEATURE_NAMES;

/// The trained model, embedded into the binary.
static MODEL_BYTES: &[u8] = include_bytes!("../assets/risk_model.onnx");
/// Learned coefficients, used to explain and (if ONNX fails) to fall back.
static COEFFICIENTS: &str = include_str!("../assets/risk_coefficients.json");

#[derive(Debug, Clone, Copy, Serialize, serde::Deserialize)]
pub struct RiskThresholds {
    /// probability >= medium is "средний риск".
    pub medium: f64,
    /// probability >= high is "высокий риск".
    pub high: f64,
    /// probability >= critical is "критический риск".
    pub critical: f64,
}

impl Default for RiskThresholds {
    /// Used only if the bundled JSON has no `risk_thresholds` block.
    fn default() -> Self {
        Self {
            medium: 0.25,
            high: 0.50,
            critical: 0.75,
        }
    }
}

/// Dataset-level importance of a feature for the exported model
/// (permutation importance: drop in average precision when shuffled).
#[derive(Debug, Clone, Serialize, serde::Deserialize)]
pub struct GlobalImportance {
    pub feature: String,
    pub importance: f64,
}

#[derive(Debug, Clone, Serialize, serde::Deserialize)]
pub struct Coefficients {
    pub feature_names: Vec<String>,
    pub weights: Vec<f64>,
    pub bias: f64,
    /// True when the exported graph is not the linear model, so the
    /// contributions below are only indicative.
    #[serde(default)]
    pub approx_explain: bool,
    /// Score cut-offs between risk levels, derived on the held-out test
    /// split by `assets/train_risk_model.py` (see `assets/train_metrics.json`
    /// for precision/recall at each cut-off).
    #[serde(default)]
    pub risk_thresholds: RiskThresholds,
    /// Which features the exported model really relies on, measured on the
    /// held-out split. Sorted by importance, descending.
    #[serde(default)]
    pub global_importance: Vec<GlobalImportance>,
}

/// Parsed once: the bundled file is a compile-time constant.
fn coefficients() -> &'static Coefficients {
    static CACHE: std::sync::OnceLock<Coefficients> = std::sync::OnceLock::new();
    CACHE.get_or_init(|| {
        serde_json::from_str(COEFFICIENTS).expect("bundled coefficients JSON is valid")
    })
}

/// Score cut-offs that separate the risk levels shown to the user.
pub fn risk_thresholds() -> RiskThresholds {
    coefficients().risk_thresholds
}

/// Features the exported model actually relies on (held-out split).
pub fn global_importance() -> &'static [GlobalImportance] {
    &coefficients().global_importance
}

#[derive(Debug, Clone, Serialize)]
pub struct Contribution {
    pub feature: String,
    pub value: f32,
    pub weight: f64,
    /// weight * value: contribution to log-odds.
    pub impact: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct RiskResult {
    /// Probability of data-loss failure, 0.0..=1.0.
    pub probability: f64,
    /// `onnx` or `heuristic` (fallback when ONNX Runtime fails).
    pub engine: String,
    pub error: Option<String>,
    /// Probability of the linear surrogate whose coefficients explain this
    /// result. Equals `probability` when the linear model is exported;
    /// otherwise a large gap means the contributions below are indicative.
    pub proxy_probability: f64,
    /// True when the exported graph is not the linear model.
    pub approx_explain: bool,
    /// All contributions sorted by |impact|, descending.
    pub contributions: Vec<Contribution>,
}

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

/// Run the embedded ONNX model on a batch of `[8]` feature vectors.
///
/// A single session is created for the whole batch, which keeps the test
/// suite (and any future batch use) cheap.
fn onnx_probabilities(batch: &[[f32; 8]]) -> Result<Vec<f64>, String> {
    let mut session = ort::session::Session::builder()
        .map_err(|e| e.to_string())?
        .commit_from_memory(MODEL_BYTES)
        .map_err(|e| e.to_string())?;
    let flat: Vec<f32> = batch.iter().flatten().copied().collect();
    let input =
        ort::value::Tensor::from_array(([batch.len(), 8], flat)).map_err(|e| e.to_string())?;
    let outputs = session
        .run(ort::inputs!["input" => input])
        .map_err(|e| e.to_string())?;
    let out = outputs
        .get("probability")
        .ok_or_else(|| "model has no 'probability' output".to_string())?;
    let arr = out.try_extract_array::<f32>().map_err(|e| e.to_string())?;
    let vals: Vec<f32> = arr.iter().copied().collect();
    // With zipmap=false the output is [p_class0, p_class1] per row.
    let stride = match arr.shape() {
        [_, cols] if *cols >= 2 => *cols,
        _ => 1,
    };
    Ok((0..batch.len())
        .map(|i| {
            let base = i * stride;
            if stride >= 2 {
                vals[base + 1] as f64
            } else {
                vals[base] as f64
            }
        })
        .collect())
}

/// Run the embedded ONNX model on a single `[1, 8]` input.
fn onnx_probability(feats: &[f32; 8]) -> Result<f64, String> {
    onnx_probabilities(std::slice::from_ref(feats)).map(|p| p[0])
}

/// Predict data-loss risk and explain it with coefficient contributions.
pub fn predict(feats: &[f32; 8]) -> RiskResult {
    let coeffs = coefficients();

    // Linear surrogate: used for the fallback engine and, whenever the
    // exported model is not the linear one, as the source of the displayed
    // contributions (with `proxy_probability` showing how far it is).
    let mut logit = coeffs.bias;
    for (w, x) in coeffs.weights.iter().zip(feats.iter()) {
        logit += w * (*x as f64);
    }
    let proxy_probability = sigmoid(logit).clamp(0.0, 1.0);

    let (probability, engine, error) = match onnx_probability(feats) {
        Ok(p) => (p.clamp(0.0, 1.0), "onnx".to_string(), None),
        Err(e) => (proxy_probability, "heuristic".to_string(), Some(e)),
    };

    let mut contributions: Vec<Contribution> = FEATURE_NAMES
        .iter()
        .zip(feats.iter())
        .zip(coeffs.weights.iter())
        .map(|((&name, &value), &weight)| Contribution {
            feature: name.to_string(),
            value,
            weight,
            impact: weight * value as f64,
        })
        .collect();
    contributions.sort_by(|a, b| {
        b.impact
            .abs()
            .partial_cmp(&a.impact.abs())
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    RiskResult {
        probability,
        engine,
        error,
        proxy_probability,
        approx_explain: coeffs.approx_explain,
        contributions,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const FIXTURE: &str = include_str!("../tests/fixtures/features_real.json");

    #[derive(serde::Deserialize)]
    struct Fixture {
        feature_names: Vec<String>,
        samples: Vec<Sample>,
    }

    #[derive(serde::Deserialize, Clone)]
    struct Sample {
        label: u8,
        features: Vec<f32>,
    }

    fn fixture() -> Fixture {
        serde_json::from_str(FIXTURE).expect("fixture parses")
    }

    fn by_label(fx: &Fixture, label: u8) -> Vec<&Sample> {
        fx.samples.iter().filter(|s| s.label == label).collect()
    }

    fn to_vec(s: &Sample) -> [f32; 8] {
        let mut f = [0f32; 8];
        f.copy_from_slice(&s.features[..8]);
        f
    }

    fn probs(samples: &[&Sample]) -> Vec<f64> {
        let batch: Vec<[f32; 8]> = samples.iter().map(|s| to_vec(s)).collect();
        onnx_probabilities(&batch).expect("embedded ONNX model runs")
    }

    fn mean(v: &[f64]) -> f64 {
        v.iter().sum::<f64>() / v.len() as f64
    }

    fn quantile(v: &mut [f64], q: f64) -> f64 {
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        v[((v.len() - 1) as f64 * q).round() as usize]
    }

    #[test]
    fn bundled_coefficients_have_expected_shape() {
        let c: Coefficients = serde_json::from_str(COEFFICIENTS).unwrap();
        assert_eq!(c.feature_names.len(), 8);
        assert_eq!(c.weights.len(), 8);
        assert_eq!(c.feature_names, FEATURE_NAMES.map(String::from).to_vec());
    }

    /// The risk-level cut-offs must be the data-derived ones, not the
    /// fallback constants in `RiskThresholds::default`.
    #[test]
    fn bundled_risk_thresholds_are_present() {
        assert!(
            COEFFICIENTS.contains("\"risk_thresholds\""),
            "assets/risk_coefficients.json has no risk_thresholds block; \
             retrain with assets/train_risk_model.py"
        );
        let t = risk_thresholds();
        assert!(
            0.0 < t.medium && t.medium < t.high && t.high < t.critical && t.critical <= 1.0,
            "cut-offs must be ordered inside (0, 1]: {t:?}"
        );
    }

    /// The features the exported model actually relies on must be bundled
    /// and sorted, so the report has something honest to show.
    #[test]
    fn bundled_global_importance_is_present() {
        let gi = global_importance();
        assert_eq!(gi.len(), FEATURE_NAMES.len(), "one entry per feature");
        assert!(
            gi.iter()
                .all(|g| g.importance.is_finite() && g.importance >= 0.0)
        );
        assert!(gi.windows(2).all(|w| w[0].importance >= w[1].importance));
        assert!(gi.iter().any(|g| g.feature == "pct_used"));
    }

    /// The surrogate probability must always be reported, and must equal
    /// the score itself when the linear model is the exported one.
    #[test]
    fn proxy_probability_matches_engine() {
        let fx = fixture();
        let sample = to_vec(fx.samples.first().expect("fixture has samples"));
        let r = predict(&sample);
        assert!((0.0..=1.0).contains(&r.proxy_probability));
        assert_eq!(r.approx_explain, coefficients().approx_explain);
        if !r.approx_explain {
            assert!((r.probability - r.proxy_probability).abs() < 1e-9);
        }
    }

    /// The fixture holds only rows from the training script's test split,
    /// so this fails if the embedded model stops working or degrades badly.
    #[test]
    fn real_holdout_samples_separate() {
        let fx = fixture();
        assert_eq!(fx.feature_names, FEATURE_NAMES.map(String::from).to_vec());
        let pos = probs(&by_label(&fx, 1));
        let neg = probs(&by_label(&fx, 0));
        let (mp, mn) = (mean(&pos), mean(&neg));
        println!("mean p(pos)={:.4} mean p(neg)={:.4}", mp, mn);

        assert!(mn < 0.10, "healthy mean risk {mn} too high");
        assert!(
            mp > 0.15,
            "pre-failure mean risk {mp} too low (is the model embedded?)"
        );
        assert!(mp > 1.5 * mn, "poor separation p(pos)={mp} p(neg)={mn}");
    }

    /// The top of the positive distribution should clear the healthy tail.
    #[test]
    fn pre_failure_tail_beats_healthy_tail() {
        let fx = fixture();
        let mut pos = probs(&by_label(&fx, 1));
        let mut neg = probs(&by_label(&fx, 0));
        let n90 = quantile(&mut neg, 0.90);
        let p75 = quantile(&mut pos, 0.75);
        println!("p75(pos)={:.4} p90(neg)={:.4}", p75, n90);
        assert!(
            p75 > n90,
            "positive tail {p75} should exceed healthy tail {n90}"
        );
    }

    /// End-to-end: the embedded graph really is executed by ONNX Runtime.
    #[test]
    fn predict_uses_onxx_engine() {
        let fx = fixture();
        let healthy = by_label(&fx, 0);
        let r = predict(&to_vec(healthy[0]));
        assert_eq!(r.engine, "onnx", "error: {:?}", r.error);
        assert!(r.error.is_none());
        assert!((0.0..=1.0).contains(&r.probability));
    }

    /// The explanation must stay well-formed even though the exported model
    /// is a boosted-tree ensemble explained by logistic coefficients.
    #[test]
    fn contributions_are_sorted_by_impact() {
        let fx = fixture();
        let pos = by_label(&fx, 1);
        let r = predict(&to_vec(pos[0]));
        assert_eq!(r.contributions.len(), 8);
        let names: Vec<&str> = r.contributions.iter().map(|c| c.feature.as_str()).collect();
        let mut sorted = names.clone();
        sorted.sort_unstable();
        let mut expected: Vec<&str> = FEATURE_NAMES.to_vec();
        expected.sort_unstable();
        assert_eq!(sorted, expected);
        for pair in r.contributions.windows(2) {
            assert!(
                pair[0].impact.abs() >= pair[1].impact.abs(),
                "contributions not sorted: {:?}",
                r.contributions
            );
        }
    }
}
