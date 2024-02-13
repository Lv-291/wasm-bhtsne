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