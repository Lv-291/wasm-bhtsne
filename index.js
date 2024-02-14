import { threads } from 'wasm-feature-detect';

function createRandomMatrix(rows, columns) {
    return Array.from({ length: rows }, () =>
        Array.from({ length: columns }, () => Math.random())
    );
}

(async function initMultiThread() {
  if (!(await threads())) return;
  const multiThread = await import('./pkg-parallel/wasm_bhtsne.js');
  await multiThread.default();
  await multiThread.initThreadPool(navigator.hardwareConcurrency);

  Object.assign(document.getElementById("wasm-bhtsne"), {
    async onclick() {
      const timeOutput = /** @type {HTMLOutputElement} */ (
          document.getElementById('time')
      );

      // create random points and dimensions
      const data = createRandomMatrix(5000, 60);

      let tsne_encoder = new multiThread.tSNE(data);
      tsne_encoder.perplexity = 10.0;

      const start = performance.now();
      let compressed_vectors;
      for (let i = 0; i < 1000; i++) {
        compressed_vectors = tsne_encoder.barnes_hut(1);
      }
      const time = performance.now() - start;

      timeOutput.value = `${time.toFixed(2)} ms`;
      console.log("Compressed Vectors:", compressed_vectors);
    },
    disabled: false
  });
})();

