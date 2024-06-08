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
  if (await threads()) {
    await multiThread.initThreadPool(navigator.hardwareConcurrency);
  }
  Object.assign(document.getElementById("wasm-bhtsne"), {
    async onclick() {
      const timeOutput = /** @type {HTMLOutputElement} */ (
          document.getElementById('time')
      );

      const opt = {
        learning_rate: 150.0,
        perplexity: 30.0,
        theta: 0.6
      };

      // create random points and dimensions
      const data = createRandomMatrix(5000, 512);

      const start = performance.now();
      let tsne_encoder = new multiThread.bhtSNEf32(data, opt);

      let compressed_vectors;
      for (let i = 0; i < 1000; i++) {
        compressed_vectors = tsne_encoder.step(1)
        /* …do something with `compressed_vectors`… */
      }

      const time = performance.now() - start;

      timeOutput.value = `${time.toFixed(2)} ms`;
      console.log("Compressed Vectors:", compressed_vectors);
    },
    disabled: false
  });
})();

