<div align="center"> <h1 align="center"> wasm-bhtsne </h1> </div>

<div align="center">

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
</div>


This is the wasm version of the [bhtsne](https://github.com/frjnn/bhtsne) crate.

## Features
- Harnesses multi-threading capabilities through [wasm-bindgen-rayon](https://github.com/RReverser/wasm-bindgen-rayon).

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

            let tsne_encoder = new multiThread.tSNE(data);
            tsne_encoder.perplexity = 10.0;

            let compressed_vectors = tsne_encoder.barnes_hut(1000);
            console.log("Compressed Vectors:", compressed_vectors);
        },
        disabled: false
    });
})();
```
