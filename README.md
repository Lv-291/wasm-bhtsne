<div align="center"> <h1 align="center"> wasm-bhtsne </h1> </div>

<div align="center">

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
</div>


This is the wasm version of the [bhtsne](https://github.com/frjnn/bhtsne) crate.

## Features
- Harnesses multi-threading capabilities through [wasm-bindgen-rayon](https://github.com/RReverser/wasm-bindgen-rayon).

## Requirements
To use the multithreading feature, you need to enable `SharedArrayBuffer` on the Web. As stated in the [wasm-bindgen-rayon readme](https://github.com/RReverser/wasm-bindgen-rayon/blob/main/README.md):

In order to use `SharedArrayBuffer` on the Web, you need to enable [cross-origin isolation policies](https://web.dev/coop-coep/). Check out the linked article for details.

## Installation
```shell
npm i wasm-bhtsne
```

## Example

```javascript
import { threads } from 'wasm-feature-detect';

function createRandomMatrix(rows, columns) {
    return Array.from({ length: rows }, () =>
        Array.from({ length: columns }, () => Math.random())
    );
}

(async function initMultiThread() {
    const multiThread = await import('./pkg-parallel/wasm_bhtsne.js');
    await multiThread.default();
    if (await threads()) {
        console.log("Browser supports threads");
        await multiThread.initThreadPool(navigator.hardwareConcurrency);
    } else {
        console.log("Browser does not support threads");
    }

    Object.assign(document.getElementById("wasm-bhtsne"), {
        async onclick() {

            // create random points and dimensions
            const data = createRandomMatrix(500, 7);

            let tsne_encoder = new multiThread.bhtSNE(data); // create a tSNE instance
            tsne_encoder.perplexity = 25.0;  // change hyperparameters

            // run the algorithm with 1000 iterations 
            let compressed_vectors = tsne_encoder.step(1000); 
            console.log("Compressed Vectors:", compressed_vectors);
        },
        disabled: false
    });
})();
```
