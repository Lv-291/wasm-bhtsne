<div align="center"> <h1 align="center"> wasm-bhtsne </h1> </div>

<div align="center">

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

</div>

This is the wasm version of the [bhtsne](https://github.com/frjnn/bhtsne) crate.

Parallel implementations of Barnes-Hut and exact implementations of the t-SNE algorithm written in Rust to run in wasm. The tree-accelerated version of the algorithm is described with fine detail in [this paper](http://lvdmaaten.github.io/publications/papers/JMLR_2014.pdf) by [Laurens van der Maaten](https://github.com/lvdmaaten). The exact, original, version of the algorithm is described in [this other paper](https://www.jmlr.org/papers/volume9/vandermaaten08a/vandermaaten08a.pdf) by [G. Hinton](https://www.cs.toronto.edu/~hinton/) and Laurens van der Maaten.
Additional implementations of the algorithm, are listed at [this page](http://lvdmaaten.github.io/tsne/).

### Example

```javascript
import {tSNE} from "wasm-bhtsne";

const data = [];

for (let i = 0; i < 300; i++) {
    const randomNumber = Math.random();
    data.push(randomNumber);
}

const tsne_encoder = new tSNE(data, 4);
tsne_encoder.exact();
const embedded_stuff = tsne_encoder.embedding();

console.log(embedded_stuff);
```

## Parallelism 
Being built on [rayon](https://github.com/rayon-rs/rayon), the algorithm uses the same number of threads as the number of CPUs available. Do note that on systems with hyperthreading enabled this equals the number of logical cores and not the physical ones. See [rayon's FAQs](https://github.com/rayon-rs/rayon/blob/master/FAQ.md) for additional informations.