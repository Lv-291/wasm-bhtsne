<div align="center"> <h1 align="center"> wasm-bhtsne </h1> </div>

<div align="center">

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
</div>


This is the wasm version of the [bhtsne](https://github.com/frjnn/bhtsne) crate.

## Features
- Harnesses multi-threading capabilities through [wasm-bindgen-rayon](https://github.com/RReverser/wasm-bindgen-rayon).
- Allows passing t-SNE hyperparameters through a JavaScript object, where you only need to include the parameters you want to change from the defaults. If you don't specify any, default values are used.
- Supports running the algorithm in iterations, enabling progressive refinement of the embedding
- Supports both Float32Array and Float64Array for data input 

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
            const data = createRandomMatrix(5000, 512);

            // Example of setting hyperparameters
            const opt = {
                learning_rate: 150.0,
                perplexity: 30.0,
                theta: 0.6
            };
            
            // let tsne_encoder = new multiThread.bhtSNEf64(data, opt);
            // or
            let tsne_encoder = new multiThread.bhtSNEf32(data, opt);
            let compressed_vectors;

            for (let i = 0; i < 1000; i++) {
                compressed_vectors = tsne_encoder.step(1)
                /* …do something with `compressed_vectors`… */
            }

            console.log("Compressed Vectors:", compressed_vectors);
        },
        disabled: false
    });
})();
```

## Hyperparameters
Here is a list of hyperparameters that can be set in the JavaScript object, along with their default values and descriptions:

- **`learning_rate`** (default: `200.0`): controls the step size during the optimization.
- **`momentum`** (default: `0.5`): helps accelerate gradients vectors in the right directions, thus leading to faster converging.
- **`final_momentum`** (default: `0.8`): momentum value used after a certain number of iterations.
- **`momentum_switch_epoch`** (default: `250`): the epoch after which the algorithm switches to `final_momentum` for the map update.
- **`stop_lying_epoch`** (default: `250`): the epoch after which the P distribution values become true. For epochs < `stop_lying_epoch`, the values of the P distribution are multiplied by a factor equal to `12.0`.
- **`theta`** (default: `0.5`): Determines the accuracy of the approximation. Larger values increase the speed but decrease accuracy. Must be strictly greater than 0.0.
- **`embedding_dim`** (default: `2`): the dimensionality of the embedding space.
- **`perplexity`** (default: `20.0`): the perplexity value. It determines the balance between local and global aspects of the data. A good value lies between 5.0 and 50.0.
    
    
    
