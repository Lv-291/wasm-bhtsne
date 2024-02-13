<div align="center"> <h1 align="center"> wasm-bhtsne </h1> </div>

<div align="center">

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

</div>

This is the wasm version of the [bhtsne](https://github.com/frjnn/bhtsne) crate.

Parallel implementations of Barnes-Hut and exact implementations of the t-SNE algorithm written in Rust to run in wasm. The tree-accelerated version of the algorithm is described with fine detail in [this paper](http://lvdmaaten.github.io/publications/papers/JMLR_2014.pdf) by [Laurens van der Maaten](https://github.com/lvdmaaten). The exact, original, version of the algorithm is described in [this other paper](https://www.jmlr.org/papers/volume9/vandermaaten08a/vandermaaten08a.pdf) by [G. Hinton](https://www.cs.toronto.edu/~hinton/) and Laurens van der Maaten.
Additional implementations of the algorithm, are listed at [this page](http://lvdmaaten.github.io/tsne/).

## Installation
```shell
npm i wasm-bhtsne
```

### Example

```javascript
import init, { initThreadPool, tSNE } from "./pkg-parallel/wasm_bhtsne.js";

await init();

//await initThreadPool(navigator.hardwareConcurrency);

function createRandomMatrix(rows, columns) {
    return Array.from({ length: rows }, () =>
        Array.from({ length: columns }, () => Math.random())
    );
}

const timeOutput = /** @type {HTMLOutputElement} */ (
    document.getElementById('time')
);

// create random points and dimensions
const data = createRandomMatrix(500, 4);

const tsne_encoder = new tSNE(data);
tsne_encoder.perplexity = 10.0;

const start = performance.now();
const compressed_vectors = tsne_encoder.barnes_hut(1000);
const time = performance.now() - start;

timeOutput.value = `${time.toFixed(2)} ms`;

console.log("Compressed Vectors:", compressed_vectors);
```