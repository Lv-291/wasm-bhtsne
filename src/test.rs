use rayon::iter::{IntoParallelIterator, ParallelIterator};

use super::{bhtSNEf32, tsne};
use wasm_bindgen::JsValue;

extern crate wasm_bindgen_test;
use crate::hyperparameters::Hyperparameters;
use wasm_bindgen_test::*;

wasm_bindgen_test_configure!(run_in_browser);
const D: usize = 4;

const NO_DIMS: u8 = 2;

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

    let opt: Hyperparameters<f32> = Hyperparameters {
        learning_rate: 200.0,
        momentum: 0.5,
        final_momentum: 0.8,
        momentum_switch_epoch: 250,
        stop_lying_epoch: 250,
        theta: 0.5,
        embedding_dim: 2,
        perplexity: 20.0,
    };

    let opt_js: JsValue = serde_wasm_bindgen::to_value(&opt).unwrap();

    let mut tsne: bhtSNEf32 = bhtSNEf32::new(data_js, opt_js);

    for _x in 0..1000 {
        tsne.step(1).unwrap();
    }
    let embedding_js = tsne.step(1).unwrap();
    let embedding_rs: Vec<Vec<f32>> = serde_wasm_bindgen::from_value(embedding_js).unwrap();
    let flattened_array: Vec<f32> = embedding_rs
        .into_par_iter()
        .flat_map(|inner_vec| inner_vec.into_par_iter())
        .collect();
    let points: Vec<_> = flattened_array.chunks(NO_DIMS as usize).collect();

    assert_eq!(points.len(), samples.len());

    assert!(
        tsne::evaluate_error_approximately(
            &tsne.tsne_encoder.p_rows,
            &tsne.tsne_encoder.p_columns,
            &tsne.tsne_encoder.p_values,
            &tsne.tsne_encoder.y,
            &samples.len(),
            &(tsne.tsne_encoder.embedding_dim),
            &tsne.tsne_encoder.theta,
        ) < 5.0
    );
}
