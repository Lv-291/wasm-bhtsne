pub fn set_panic_hook() {
    // When the `console_error_panic_hook` feature is enabled, we can call the
    // `set_panic_hook` function at least once during initialization, and then
    // we will get better error messages if our code ever panics.
    //
    // For more details see
    // https://github.com/rustwasm/console_error_panic_hook#readme
    #[cfg(feature = "console_error_panic_hook")]
    console_error_panic_hook::set_once();
}
use js_sys::{Array, Float32Array};
use wasm_bindgen::prelude::*;

pub fn convert_to_array_of_arrays(input_data: Vec<f32>, num_cols: usize) -> Array {
    let js_result_matrix = Array::new();

    for i in 0..input_data.len() / num_cols {
        let start_index = i * num_cols;
        let end_index = start_index + num_cols;
        let row_slice = &input_data[start_index..end_index];
        let js_row = Float32Array::from(row_slice);
        js_result_matrix.push(&js_row);
    }

    js_result_matrix
}

pub fn convert_array_to_vec(input_array: &Array) -> Vec<f32> {
    let mut output_vec = Vec::new();

    for i in 0..input_array.length() {
        if let Ok(row) = input_array.get(i).dyn_into::<Array>() {
            for j in 0..row.length() {
                if let Some(value) = row.get(j).as_f64() {
                    output_vec.push(value as f32);
                }
            }
        }
    }
    output_vec
}

pub fn get_num_cols(input_array: &Array) -> usize {
    let first_row = input_array.get(0).dyn_into::<Array>().unwrap();
    first_row.length() as usize
}
