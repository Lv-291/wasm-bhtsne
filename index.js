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

      // create random points and dimensions
      const data = createRandomMatrix(5000, 512);


      const start = performance.now();
      let tsne_encoder = new multiThread.bhtSNE(data);
      //tsne_encoder.learning_rate()

      let compressed_vectors = tsne_encoder.step();
      const time = performance.now() - start;
      tsne_encoder.free();

      timeOutput.value = `${time.toFixed(2)} ms`;
      console.log("Compressed Vectors:", compressed_vectors);
    },
    disabled: false
  });
})();

