use super::{bhtSNE, tsne};
use wasm_bindgen::JsValue;
use wasm_bindgen_test::wasm_bindgen_test;
use wasm_bindgen_test::wasm_bindgen_test_configure;

wasm_bindgen_test_configure!(run_in_browser);

const D: usize = 4;
const THETA: f32 = 0.5;
const NO_DIMS: u8 = 2;

#[wasm_bindgen_test]
#[cfg(not(tarpaulin_include))]
fn set_learning_rate() {
    let data_rs: Vec<Vec<f32>> = vec![vec![0.]];
    let data_js: JsValue = serde_wasm_bindgen::to_value(&data_rs).unwrap();
    let mut tsne: bhtSNE = bhtSNE::new(data_js);
    tsne.learning_rate(15.);
    assert_eq!(tsne.tsne_encoder.learning_rate, 15.);
}

#[wasm_bindgen_test]
#[cfg(not(tarpaulin_include))]
fn set_epochs() {
    let data_rs: Vec<Vec<f32>> = vec![vec![0.]];
    let data_js: JsValue = serde_wasm_bindgen::to_value(&data_rs).unwrap();
    let mut tsne: bhtSNE = bhtSNE::new(data_js);
    tsne.epochs(15);
    assert_eq!(tsne.tsne_encoder.epochs, 15);
}

#[wasm_bindgen_test]
#[cfg(not(tarpaulin_include))]
fn set_momentum() {
    let data_rs: Vec<Vec<f32>> = vec![vec![0.]];
    let data_js: JsValue = serde_wasm_bindgen::to_value(&data_rs).unwrap();
    let mut tsne: bhtSNE = bhtSNE::new(data_js);
    tsne.momentum(15.);
    assert_eq!(tsne.tsne_encoder.momentum, 15.);
}

#[wasm_bindgen_test]
#[cfg(not(tarpaulin_include))]
fn set_final_momentum() {
    let data_rs: Vec<Vec<f32>> = vec![vec![0.]];
    let data_js: JsValue = serde_wasm_bindgen::to_value(&data_rs).unwrap();
    let mut tsne: bhtSNE = bhtSNE::new(data_js);
    tsne.final_momentum(15.);
    assert_eq!(tsne.tsne_encoder.final_momentum, 15.);
}

#[wasm_bindgen_test]
#[cfg(not(tarpaulin_include))]
fn set_momentum_switch_epoch() {
    let data_rs: Vec<Vec<f32>> = vec![vec![0.]];
    let data_js: JsValue = serde_wasm_bindgen::to_value(&data_rs).unwrap();
    let mut tsne: bhtSNE = bhtSNE::new(data_js);
    tsne.momentum_switch_epoch(15);
    assert_eq!(tsne.tsne_encoder.momentum_switch_epoch, 15);
}

#[wasm_bindgen_test]
#[cfg(not(tarpaulin_include))]
fn set_stop_lying_epoch() {
    let data_rs: Vec<Vec<f32>> = vec![vec![0.]];
    let data_js: JsValue = serde_wasm_bindgen::to_value(&data_rs).unwrap();
    let mut tsne: bhtSNE = bhtSNE::new(data_js);
    tsne.stop_lying_epoch(15);
    assert_eq!(tsne.tsne_encoder.stop_lying_epoch, 15);
}

#[wasm_bindgen_test]
#[cfg(not(tarpaulin_include))]
fn set_embedding_dim() {
    let data_rs: Vec<Vec<f32>> = vec![vec![0.]];
    let data_js: JsValue = serde_wasm_bindgen::to_value(&data_rs).unwrap();
    let mut tsne: bhtSNE = bhtSNE::new(data_js);
    tsne.embedding_dim(3);
    assert_eq!(tsne.tsne_encoder.embedding_dim, 3);
}

#[wasm_bindgen_test]
#[cfg(not(tarpaulin_include))]
fn set_perplexity() {
    let data_rs: Vec<Vec<f32>> = vec![vec![0.]];
    let data_js: JsValue = serde_wasm_bindgen::to_value(&data_rs).unwrap();
    let mut tsne: bhtSNE = bhtSNE::new(data_js);
    tsne.perplexity(15.);
    assert_eq!(tsne.tsne_encoder.perplexity, 15.);
}

#[wasm_bindgen_test]
#[cfg(not(tarpaulin_include))]
fn barnes_hut_tsne() {

    // TODO: implementing I/O for testing with iris.csv and maybe a pkg feature

    // for now this don't work, wasm doesn't support I/O out of the box
    //   let data: Vec<f32> =
    //      crate::load_csv("iris.csv", true, Some(&[4]), |float| float.parse().unwrap()).unwrap();

    // this is ugly but i kinda like it
    let data: Vec<f32> = vec![
        5.1, 3.5, 1.4, 0.2, 4.9, 3., 1.4, 0.2, 4.7, 3.2, 1.3, 0.2, 4.6, 3.1, 1.5, 0.2, 5., 3.6,
        1.4, 0.2, 5.4, 3.9, 1.7, 0.4, 4.6, 3.4, 1.4, 0.3, 5., 3.4, 1.5, 0.2, 4.4, 2.9, 1.4, 0.2,
        4.9, 3.1, 1.5, 0.1, 5.4, 3.7, 1.5, 0.2, 4.8, 3.4, 1.6, 0.2, 4.8, 3., 1.4, 0.1, 4.3, 3.,
        1.1, 0.1, 5.8, 4., 1.2, 0.2, 5.7, 4.4, 1.5, 0.4, 5.4, 3.9, 1.3, 0.4, 5.1, 3.5, 1.4, 0.3,
        5.7, 3.8, 1.7, 0.3, 5.1, 3.8, 1.5, 0.3, 5.4, 3.4, 1.7, 0.2, 5.1, 3.7, 1.5, 0.4, 4.6, 3.6,
        1., 0.2, 5.1, 3.3, 1.7, 0.5, 4.8, 3.4, 1.9, 0.2, 5., 3., 1.6, 0.2, 5., 3.4, 1.6, 0.4, 5.2,
        3.5, 1.5, 0.2, 5.2, 3.4, 1.4, 0.2, 4.7, 3.2, 1.6, 0.2, 4.8, 3.1, 1.6, 0.2, 5.4, 3.4, 1.5,
        0.4, 5.2, 4.1, 1.5, 0.1, 5.5, 4.2, 1.4, 0.2, 4.9, 3.1, 1.5, 0.2, 5., 3.2, 1.2, 0.2, 5.5,
        3.5, 1.3, 0.2, 4.9, 3.6, 1.4, 0.1, 4.4, 3., 1.3, 0.2, 5.1, 3.4, 1.5, 0.2, 5., 3.5, 1.3,
        0.3, 4.5, 2.3, 1.3, 0.3, 4.4, 3.2, 1.3, 0.2, 5., 3.5, 1.6, 0.6, 5.1, 3.8, 1.9, 0.4, 4.8,
        3., 1.4, 0.3, 5.1, 3.8, 1.6, 0.2, 4.6, 3.2, 1.4, 0.2, 5.3, 3.7, 1.5, 0.2, 5., 3.3, 1.4,
        0.2, 7., 3.2, 4.7, 1.4, 6.4, 3.2, 4.5, 1.5, 6.9, 3.1, 4.9, 1.5, 5.5, 2.3, 4., 1.3, 6.5,
        2.8, 4.6, 1.5, 5.7, 2.8, 4.5, 1.3, 6.3, 3.3, 4.7, 1.6, 4.9, 2.4, 3.3, 1., 6.6, 2.9, 4.6,
        1.3, 5.2, 2.7, 3.9, 1.4, 5., 2., 3.5, 1., 5.9, 3., 4.2, 1.5, 6., 2.2, 4., 1., 6.1, 2.9,
        4.7, 1.4, 5.6, 2.9, 3.6, 1.3, 6.7, 3.1, 4.4, 1.4, 5.6, 3., 4.5, 1.5, 5.8, 2.7, 4.1, 1.,
        6.2, 2.2, 4.5, 1.5, 5.6, 2.5, 3.9, 1.1, 5.9, 3.2, 4.8, 1.8, 6.1, 2.8, 4., 1.3, 6.3, 2.5,
        4.9, 1.5, 6.1, 2.8, 4.7, 1.2, 6.4, 2.9, 4.3, 1.3, 6.6, 3., 4.4, 1.4, 6.8, 2.8, 4.8, 1.4,
        6.7, 3., 5., 1.7, 6., 2.9, 4.5, 1.5, 5.7, 2.6, 3.5, 1., 5.5, 2.4, 3.8, 1.1, 5.5, 2.4, 3.7,
        1., 5.8, 2.7, 3.9, 1.2, 6., 2.7, 5.1, 1.6, 5.4, 3., 4.5, 1.5, 6., 3.4, 4.5, 1.6, 6.7, 3.1,
        4.7, 1.5, 6.3, 2.3, 4.4, 1.3, 5.6, 3., 4.1, 1.3, 5.5, 2.5, 4., 1.3, 5.5, 2.6, 4.4, 1.2,
        6.1, 3., 4.6, 1.4, 5.8, 2.6, 4., 1.2, 5., 2.3, 3.3, 1., 5.6, 2.7, 4.2, 1.3, 5.7, 3., 4.2,
        1.2, 5.7, 2.9, 4.2, 1.3, 6.2, 2.9, 4.3, 1.3, 5.1, 2.5, 3., 1.1, 5.7, 2.8, 4.1, 1.3, 6.3,
        3.3, 6., 2.5, 5.8, 2.7, 5.1, 1.9, 7.1, 3., 5.9, 2.1, 6.3, 2.9, 5.6, 1.8, 6.5, 3., 5.8, 2.2,
        7.6, 3., 6.6, 2.1, 4.9, 2.5, 4.5, 1.7, 7.3, 2.9, 6.3, 1.8, 6.7, 2.5, 5.8, 1.8, 7.2, 3.6,
        6.1, 2.5, 6.5, 3.2, 5.1, 2., 6.4, 2.7, 5.3, 1.9, 6.8, 3., 5.5, 2.1, 5.7, 2.5, 5., 2., 5.8,
        2.8, 5.1, 2.4, 6.4, 3.2, 5.3, 2.3, 6.5, 3., 5.5, 1.8, 7.7, 3.8, 6.7, 2.2, 7.7, 2.6, 6.9,
        2.3, 6., 2.2, 5., 1.5, 6.9, 3.2, 5.7, 2.3, 5.6, 2.8, 4.9, 2., 7.7, 2.8, 6.7, 2., 6.3, 2.7,
        4.9, 1.8, 6.7, 3.3, 5.7, 2.1, 7.2, 3.2, 6., 1.8, 6.2, 2.8, 4.8, 1.8, 6.1, 3., 4.9, 1.8,
        6.4, 2.8, 5.6, 2.1, 7.2, 3., 5.8, 1.6, 7.4, 2.8, 6.1, 1.9, 7.9, 3.8, 6.4, 2., 6.4, 2.8,
        5.6, 2.2, 6.3, 2.8, 5.1, 1.5, 6.1, 2.6, 5.6, 1.4, 7.7, 3., 6.1, 2.3, 6.3, 3.4, 5.6, 2.4,
        6.4, 3.1, 5.5, 1.8, 6., 3., 4.8, 1.8, 6.9, 3.1, 5.4, 2.1, 6.7, 3.1, 5.6, 2.4, 6.9, 3.1,
        5.1, 2.3, 5.8, 2.7, 5.1, 1.9, 6.8, 3.2, 5.9, 2.3, 6.7, 3.3, 5.7, 2.5, 6.7, 3., 5.2, 2.3,
        6.3, 2.5, 5., 1.9, 6.5, 3., 5.2, 2., 6.2, 3.4, 5.4, 2.3, 5.9, 3., 5.1, 1.8,
    ];
    let samples: Vec<Vec<f32>> = data.chunks(D).map(|chunk| chunk.to_vec()).collect();
    let data_js: JsValue = serde_wasm_bindgen::to_value(&samples).unwrap();

    let mut tsne: bhtSNE = bhtSNE::new(data_js);
    tsne.step(0.5);

    let embedding = tsne.get_solution();
    let points: Vec<_> = embedding.chunks(NO_DIMS as usize).collect();

    assert_eq!(points.len(), samples.len());

    assert!(
        tsne::evaluate_error_approximately(
            &tsne.tsne_encoder.p_rows,
            &tsne.tsne_encoder.p_columns,
            &tsne.tsne_encoder.p_values,
            &tsne.tsne_encoder.y,
            &samples.len(),
            &(tsne.tsne_encoder.embedding_dim as usize),
            &THETA
        ) < 5.0
    );
}
